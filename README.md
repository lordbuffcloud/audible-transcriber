# Audible Transcriber

Full-pipeline CLI tool to transcribe Audible audiobooks to text.

## Features
- Authenticate with Audible and retrieve activation bytes
- List your Audible library with ASINs
- Download AAX files directly from Audible
- Decrypt AAX to WAV using ffmpeg + activation bytes
- Transcribe audio using OpenAI Whisper (multiple model sizes)
- Output formats: plain text, SRT subtitles, JSON

## Requirements
- Python 3.10+
- ffmpeg (system install)
- Audible account credentials

## Installation
```bash
cd /c/Users/lordb/code/audible-transcriber
./venv/Scripts/pip.exe install -r requirements.txt
```

## Usage

### 1. Authenticate
```bash
./venv/Scripts/python.exe transcriber.py auth
```
This saves auth.json and activation bytes to .env.

### 2. List your books
```bash
./venv/Scripts/python.exe transcriber.py list-books
```

### 3. Transcribe a book
```bash
./venv/Scripts/python.exe transcriber.py transcribe --asin <BOOK_ASIN> --model base --format txt
```

### 4. Transcribe existing AAX file (no download)
```bash
./venv/Scripts/python.exe transcriber.py transcribe-file --input /path/to/book.aax --model base
```

## Whisper Models
- `tiny`: fastest, least accurate
- `base`: good balance (default)
- `small`: better accuracy
- `medium`: high accuracy
- `large`: best accuracy, slowest

## Output
Transcripts saved to `./transcripts/` by default. Configure with `OUTPUT_DIR` env var.

## Notes
- Activation bytes are required for AAX decryption
- Audible API access is subject to their terms of service
- Whisper models are downloaded on first use (~75MB for base)
