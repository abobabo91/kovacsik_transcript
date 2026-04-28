from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import tomllib
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks


AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".mp4", ".mov", ".avi", ".mkv"}
CHUNK_LENGTH_MS = 10 * 60 * 1000
SAFE_DIRECT_UPLOAD_BYTES = 10 * 1024 * 1024


def load_openai_api_key(tool_root: Path) -> str:
    secrets_path = tool_root / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        raise FileNotFoundError(f"Missing secrets file: {secrets_path}")

    with secrets_path.open("rb") as fh:
        secrets = tomllib.load(fh)

    api_key = (
        secrets.get("openai", {}).get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .streamlit/secrets.toml or environment.")
    return api_key


def iter_audio_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def transcribe_chunk(client: OpenAI, chunk_path: Path, model: str, language: str | None) -> str:
    last_exc = None
    for attempt in range(1, 6):
        try:
            with chunk_path.open("rb") as fh:
                kwargs = {"model": model, "file": fh}
                if language:
                    kwargs["language"] = language
                response = client.audio.transcriptions.create(**kwargs)
            return response.text.strip()
        except Exception as exc:
            last_exc = exc
            if attempt == 5:
                break
            wait_s = min(30, attempt * 5)
            print(f"[RETRY] {chunk_path.name} attempt {attempt}/5 failed: {exc}. Waiting {wait_s}s...")
            sys.stdout.flush()
            time.sleep(wait_s)
    raise last_exc


def export_chunk(chunk: AudioSegment) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    chunk_path = Path(tmp.name)
    chunk.export(chunk_path, format="mp3", bitrate="192k")
    return chunk_path


def transcribe_file(client: OpenAI, audio_path: Path, output_path: Path, model: str, language: str | None) -> dict:
    ensure_parent(output_path)
    print(f"[START] {audio_path}")
    sys.stdout.flush()

    transcripts: list[str] = []
    used_chunking = audio_path.stat().st_size > SAFE_DIRECT_UPLOAD_BYTES

    if not used_chunking:
        try:
            text = transcribe_chunk(client, audio_path, model, language)
            transcripts.append(text)
        except Exception:
            used_chunking = True

    if used_chunking:
        audio = AudioSegment.from_file(audio_path)
        chunks = make_chunks(audio, CHUNK_LENGTH_MS)
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            chunk_path = export_chunk(chunk)
            try:
                print(f"[CHUNK] {audio_path.name} {idx}/{total}")
                sys.stdout.flush()
                text = transcribe_chunk(client, chunk_path, model, language)
                transcripts.append(text)
            finally:
                if chunk_path.exists():
                    chunk_path.unlink()

    combined = "\n\n".join(part for part in transcripts if part)
    output_path.write_text(combined, encoding="utf-8")
    print(f"[DONE]  {output_path}")
    sys.stdout.flush()
    return {
        "audio_path": str(audio_path),
        "output_path": str(output_path),
        "chunked": used_chunking,
        "chars": len(combined),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-transcribe audio files into raw text files.")
    parser.add_argument("--input-root", required=True, help="Folder to scan recursively for audio/video files.")
    parser.add_argument("--output-root", required=True, help="Folder where raw text files will be written.")
    parser.add_argument("--model", default="gpt-4o-transcribe", help="Transcription model.")
    parser.add_argument("--language", default="en", help="Language code. Use empty string for auto-detect.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip transcript files that already exist.")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    tool_root = Path(__file__).resolve().parent
    language = args.language or None

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    api_key = load_openai_api_key(tool_root)
    client = OpenAI(api_key=api_key)
    audio_files = iter_audio_files(input_root)

    if not audio_files:
        print("No audio files found.")
        return 0

    manifest: list[dict] = []
    for audio_path in audio_files:
        relative = audio_path.relative_to(input_root)
        output_path = (output_root / relative).with_suffix(".txt")

        if args.skip_existing and output_path.exists():
            print(f"[SKIP]  {output_path}")
            manifest.append({
                "audio_path": str(audio_path),
                "output_path": str(output_path),
                "skipped": True,
            })
            continue

        try:
            item = transcribe_file(client, audio_path, output_path, args.model, language)
            manifest.append(item)
        except Exception as exc:
            print(f"[FAIL]  {audio_path}: {exc}", file=sys.stderr)
            manifest.append({
                "audio_path": str(audio_path),
                "error": str(exc),
            })

    ensure_parent(output_root / "manifest.json")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Manifest written to {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
