# Deploying RATA Braille Recognition

Upload the website files directly from the `C:\Braille` root. No website asset
is kept in a subfolder, and every root-level website file is below 20 MB. The loader downloads these eight model files and joins
them in memory before creating the ONNX Runtime session:

- `model.onnx.part001` through `model.onnx.part007` — 16,777,216 bytes each
- `model.onnx.part008` — 5,687,177 bytes

Each file is below a 20 MB single-file upload limit. All eight files must remain
in the website root with exactly these names. Do not combine, rename, or
individually replace them.

The Android APK is at
`android/app/build/outputs/apk/debug/app-debug.apk`. It bundles the same four
parts and runs without downloading the model after installation. It supports
Android 6.0 (API 23) and newer, subject to the device having enough free storage
and memory for this 123 MB inference model.
