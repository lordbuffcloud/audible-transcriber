#!/usr/bin/env python3
# ⬡ CK42X · ck42x.com
# MODULE: click cli pipeline for audible books: auth, download, decrypt, transcribe
# gold on black. sting once.
import os
import sys
import click
import dotenv
from audible import Authenticator, Client
import subprocess
import whisper
import requests
import json
from pathlib import Path

dotenv.load_dotenv()

ACTIVATION_BYTES = os.getenv("AUDIBLE_ACTIVATION_BYTES")
AUTH_FILE = os.getenv("AUDIBLE_AUTH_FILE", "auth.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./transcripts")

def get_activation_bytes(auth):
    """return activation bytes for the account; any failure becomes RuntimeError."""
    try:
        from audible.auth import get_activation_bytes
        bytes_val = get_activation_bytes(auth)
        return bytes_val
    except Exception as e:
        raise RuntimeError(f"Failed to get activation bytes: {e}")

def download_aax(client, asin, output_path):
    """stream the aax to output_path and return it; url/http failures become RuntimeError."""
    click.echo(f"Downloading AAX for ASIN: {asin}")
    try:
        product = client.get_product(asin)
        content_url = product.get("content_url")
        if not content_url:
            raise ValueError("No download URL found")
        
        response = requests.get(content_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        click.echo(f"Downloaded to {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

def decrypt_aax(aax_path, output_wav, activation_bytes):
    """decrypt aax to wav via ffmpeg; return output_wav. failure raises RuntimeError with stderr."""
    click.echo(f"Decrypting {aax_path} to {output_wav}")
    cmd = [
        "ffmpeg", "-y",
        "-activation_bytes", activation_bytes,
        "-i", aax_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # WHY: whisper expects 16 khz audio
        output_wav
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        click.echo(f"Decrypted to {output_wav}")
        return output_wav
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Decryption failed: {e.stderr.decode()}")

def transcribe_audio(wav_path, model_size="base"):
    """run whisper on wav_path; return the raw result dict with text and segments."""
    click.echo(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    click.echo(f"Transcribing {wav_path}")
    result = model.transcribe(wav_path, fp16=False)
    return result

def save_transcript(result, output_path, format="txt"):
    """write result as txt, srt, or json; return the output path under OUTPUT_DIR."""
    # TRAP: parameter format shadows the builtin
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if format == "txt":
        output_file = Path(output_path).with_suffix(".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
    elif format == "srt":
        output_file = Path(output_path).with_suffix(".srt")
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                f.write(f"{i}\n")
                # TRAP: raw float seconds; srt needs HH:MM:SS,mmm and players reject this file
                f.write(f"{segment['start']:.3f} --> {segment['end']:.3f}\n")
                f.write(f"{segment['text'].strip()}\n\n")
    elif format == "json":
        output_file = Path(output_path).with_suffix(".json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # TRAP: unknown format leaves output_file unbound; this echo raises NameError
    click.echo(f"Saved transcript to {output_file}")
    return output_file

@click.group()
def cli():
    """audible audiobook transcriber cli."""
    pass

@cli.command()
@click.option("--username", prompt=True, hide_input=False)
@click.option("--password", prompt=True, hide_input=True)
@click.option("--locale", default="us", help="Audible locale (us, uk, de, etc.)")
def auth(username, password, locale):
    """log in, save credentials to AUTH_FILE, append activation bytes to .env; exit 1 on failure."""
    click.echo("Authenticating with Audible...")
    try:
        auth = Authenticator.from_login(
            username=username,
            password=password,
            locale=locale
        )
        auth.to_file(AUTH_FILE)
        click.echo(f"Saved auth to {AUTH_FILE}")
        
        activation_bytes = get_activation_bytes(auth)
        with open(".env", "a") as f:
            f.write(f"\nAUDIBLE_ACTIVATION_BYTES={activation_bytes}\n")
        click.echo(f"Saved activation bytes to .env")
    except Exception as e:
        click.echo(f"Authentication failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--asin", required=True, help="Audible ASIN of the book")
@click.option("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
@click.option("--format", default="txt", type=click.Choice(["txt", "srt", "json"]), help="Output format")
def transcribe(asin, model, format):
    """download, decrypt, transcribe one asin; exit 1 when auth or activation bytes are missing."""
    if not os.path.exists(AUTH_FILE):
        click.echo("Auth file not found. Run 'auth' command first.", err=True)
        sys.exit(1)
    
    if not ACTIVATION_BYTES:
        click.echo("Activation bytes not found. Run 'auth' command first.", err=True)
        sys.exit(1)
    
    try:
        auth = Authenticator.from_file(AUTH_FILE)
        client = Client(auth=auth)
        
        aax_path = f"./temp_{asin}.aax"
        download_aax(client, asin, aax_path)
        
        wav_path = f"./temp_{asin}.wav"
        decrypt_aax(aax_path, wav_path, ACTIVATION_BYTES)
        
        result = transcribe_audio(wav_path, model)
        
        output_base = os.path.join(OUTPUT_DIR, asin)
        save_transcript(result, output_base, format)
        
        os.remove(aax_path)
        os.remove(wav_path)
        click.echo("Done!")
        
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--input", required=True, type=click.Path(exists=True), help="Path to AAX file")
@click.option("--model", default="base", help="Whisper model size")
@click.option("--format", default="txt", type=click.Choice(["txt", "srt", "json"]))
def transcribe_file(input, model, format):
    """decrypt and transcribe a local aax; exit 1 without activation bytes."""
    if not ACTIVATION_BYTES:
        click.echo("Activation bytes not found. Run 'auth' command first.", err=True)
        sys.exit(1)
    
    try:
        wav_path = "./temp_decrypted.wav"
        decrypt_aax(input, wav_path, ACTIVATION_BYTES)
        
        result = transcribe_audio(wav_path, model)
        
        output_base = os.path.join(OUTPUT_DIR, Path(input).stem)
        save_transcript(result, output_base, format)
        
        os.remove(wav_path)
        click.echo("Done!")
        
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        sys.exit(1)

@cli.command()
def list_books():
    """print every library book; exit 1 without an auth file."""
    if not os.path.exists(AUTH_FILE):
        click.echo("Auth file not found. Run 'auth' command first.", err=True)
        sys.exit(1)
    
    auth = Authenticator.from_file(AUTH_FILE)
    client = Client(auth=auth)
    
    page = 1
    while True:
        response = client.get("library/books", params={"page": page, "per_page": 50})
        books = response.get("items", [])
        if not books:
            break
        
        for book in books:
            asin = book.get("asin")
            title = book.get("title")
            authors = ", ".join([a.get("name") for a in book.get("authors", [])])
            click.echo(f"ASIN: {asin} | {title} by {authors}")
        
        page += 1

if __name__ == "__main__":
    cli()
