# Standalone Qwen3-VL Encoder — Firebase Loop Runbook

How to build the standalone vision encoder, cross-compile it for Android, wrap
it in a game-loop APK, and run it on **Firebase Test Lab** (Pixel 9 Pro / Vulkan
and Galaxy S25 / OpenCL) — with an on-device cosine-similarity accuracy check.

The encoder (`tools/mtmd/qwen3vl-encoder/main.cpp`) runs ONLY the vision encoder
(ViT + patch-merge projection) through ggml — **no libllama, no LLM**. It links
only `ggml` + backends, so builds and iterates fast.

> Paths below are for this dev box. Fabric tree:
> `/home/olya/claude_folders/qvac-fabric-llm.cpp` (branch
> `feat/qwen3vl-standalone-encoder`). APK harness:
> `/home/olya/claude_folders/qwen3vl-encoder-apk`.

---

## 0. Assets

- q8 mmproj: `.../llm-llamacpp/test/model/mmproj-Qwen3.5-0.8B-Q8_0.gguf`
- image: `.../llm-llamacpp/test/mobile-profile/testAssets/elephant.jpg` (612×408
  → 247 tokens)

## 1. CLI

```
qwen3vl-encoder --mmproj <q8.gguf> --image <img>
                --backend cpu|vulkan|opencl|gpu|all
                [--device NAME] [--threads N] [--iters N]
                [--list-devices] [--dump out.bin] [--ref ref.bin] [--cos-min f]
```

- `--backend all` — CPU + every GPU device. **This is what the APK uses.** CPU
  runs first as the accuracy reference; every GPU backend is then cos-sim'd
  against it and prints `[ENC_COSSIM] label=.. cos=.. PASS/FAIL (vs cpu)`.
- Output lines: `[ENC_RESULT] label=.. tokens=.. avg_ms=..`,
  `[ENC_COSSIM] ..`, `[ENC_DONE] ran=N/M rc=..`.
- **Accuracy bar:** same-backend regression (`--ref`) uses `--cos-min` (default
  0.9999, ~1.0 expected). Cross-backend GPU-vs-CPU uses a built-in 0.99 (normal
  fp16 divergence ~0.995 PASS; a broken backend ~0.6 FAIL). **Token count alone
  is NOT correctness** — always read the cos.

---

## 2. Local build + verify (desktop, clang-22, Vulkan)

```bash
cd /home/olya/claude_folders/qvac-fabric-llm.cpp

cmake -S . -B build-encoder -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_MTMD=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_CURL=OFF

cmake --build build-encoder --target qwen3vl-encoder -j12   # only ggml+vulkan+enc

# CPU golden + Vulkan, with cross-backend cos-sim:
./build-encoder/bin/qwen3vl-encoder --mmproj <q8.gguf> --image <elephant.jpg> \
  --backend all --iters 5
```

Expect `tokens=247`, and Vulkan `cos ~0.9952 PASS (vs cpu)`.

---

## 3. Android cross-compile (NDK r27.2, ARM64)

Two **separate** single-backend builds (a combined binary hard-`NEEDED`s both
`libggml-vulkan.so` AND `libggml-opencl.so`; if `libOpenCL.so` can't resolve on
a device the whole binary fails to launch — losing the good backend too).

### 3a. Vulkan (for Pixel)

```bash
NDK=$ANDROID_NDK_HOME
SPIRV_INC=/usr/local/share/vcpkg/packages/spirv-headers_arm64-android/include

cmake -S . -B build-android-vk -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF \
  -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers \
  -DCMAKE_CXX_FLAGS="-I$SPIRV_INC" \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_MTMD=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_CURL=OFF

cmake --build build-android-vk --target qwen3vl-encoder -j12
```

> `-I$SPIRV_INC` is required: `ggml-vulkan.cpp` includes
> `spirv/unified1/spirv.hpp`, which is NOT in the NDK Vulkan sysroot. NDK 27.2
> already ships `vulkan.hpp` (v275); if a future NDK doesn't, fetch the matching
> Vulkan-Hpp into the NDK sysroot.

Produces `build-android-vk/bin/`: `qwen3vl-encoder`, `libggml.so`,
`libggml-base.so`, `libggml-cpu.so`, `libggml-vulkan.so` (~122 MB, embedded
shaders).

### 3b. OpenCL (for S25)

```bash
NDK=$ANDROID_NDK_HOME
CL_INC=/usr/local/share/vcpkg/packages/opencl-headers_arm64-android/include
CL_LIB=/usr/local/share/vcpkg/packages/opencl_arm64-android/lib/libOpenCL.so

cmake -S . -B build-android-cl -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF \
  -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DOpenCL_INCLUDE_DIR=$CL_INC -DOpenCL_LIBRARY=$CL_LIB -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_MTMD=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_CURL=OFF

cmake --build build-android-cl --target qwen3vl-encoder -j12
```

`GGML_OPENCL_EMBED_KERNELS=ON` bakes the `.cl` kernels in — no runtime files.
`libggml-opencl.so` hard-`NEEDED`s `libOpenCL.so`; it resolves on-device from
`/vendor/lib64` (see LD_LIBRARY_PATH in the APK). Adreno kernels auto-enable.

---

## 4. Build the game-loop APK

Firebase runs APKs, not bare binaries. The harness project is at
`/home/olya/claude_folders/qwen3vl-encoder-apk` (gradle 8.14.3 wrapper, AGP
8.6.0, JDK 21, applicationId `com.qvac.encbench`). Key mechanics:

- The encoder exe is shipped as `lib/arm64-v8a/libqwen3vl-encoder.so` in
  `jniLibs` (it's an ET_DYN PIE, so the packager accepts it as a "lib").
  `android:extractNativeLibs=true` + `useLegacyPackaging true` → it extracts to
  `nativeLibraryDir`, an exec-able mount.
- `MainActivity` (TEST_LOOP intent) copies the mmproj + image assets to
  `filesDir`, execs the binary with
  `LD_LIBRARY_PATH=nativeDir:/vendor/lib64:/system/vendor/lib64:/system/lib64`
  (the vendor dirs supply Adreno/Mali `libOpenCL.so`), and streams stdout/stderr
  to logcat under tag **`ENCBENCH`**, then `finish()`.
- Assets (mmproj, elephant.jpg) live in `app/src/main/assets/`; `noCompress
  'gguf','jpg'`.

Stage assets once:

```bash
APK=/home/olya/claude_folders/qwen3vl-encoder-apk
cp <q8.gguf>      $APK/app/src/main/assets/mmproj-Qwen3.5-0.8B-Q8_0.gguf
cp <elephant.jpg> $APK/app/src/main/assets/elephant.jpg
```

Build the **Vulkan** APK:

```bash
FAB=/home/olya/claude_folders/qvac-fabric-llm.cpp
JNI=$APK/app/src/main/jniLibs/arm64-v8a
rm -f $JNI/*.so
cp $FAB/build-android-vk/bin/qwen3vl-encoder $JNI/libqwen3vl-encoder.so
cp $FAB/build-android-vk/bin/libggml.so $FAB/build-android-vk/bin/libggml-base.so \
   $FAB/build-android-vk/bin/libggml-cpu.so $FAB/build-android-vk/bin/libggml-vulkan.so $JNI/
( cd $APK && ./gradlew :app:assembleDebug --no-daemon )
cp $APK/app/build/outputs/apk/debug/app-debug.apk $APK/encoder-vk.apk
```

Build the **OpenCL** APK (swap the jniLibs):

```bash
rm -f $JNI/*.so
cp $FAB/build-android-cl/bin/qwen3vl-encoder $JNI/libqwen3vl-encoder.so
cp $FAB/build-android-cl/bin/libggml.so $FAB/build-android-cl/bin/libggml-base.so \
   $FAB/build-android-cl/bin/libggml-cpu.so $FAB/build-android-cl/bin/libggml-opencl.so $JNI/
( cd $APK && ./gradlew :app:assembleDebug --no-daemon )
cp $APK/app/build/outputs/apk/debug/app-debug.apk $APK/encoder-cl.apk
```

Sanity: `unzip -l encoder-vk.apk | grep vulkan` (1) and
`unzip -l encoder-cl.apk | grep -c vulkan` (0). Debug-signed APKs are accepted
by Firebase game-loop.

---

## 5. Run on Firebase Test Lab

Devices: **Pixel 9 Pro = `caiman`** (v34/35, Vulkan), **Galaxy S25 = `pa1q`**
(v36, OpenCL). (S25 Ultra=`pa3q`, S25+=`pa2q`.) Account `olya.sirkin@tether.to`,
project `qvac-test`. Use `--async` — synchronous runs exceed the shell timeout,
and backgrounded blocking submits get killed.

```bash
GC=/home/olya/google-cloud-sdk/bin/gcloud
cd $APK

# Pixel 9 Pro / Vulkan
$GC firebase test android run --type game-loop --app encoder-vk.apk \
  --device model=caiman,version=34,locale=en,orientation=portrait \
  --timeout 600s --project qvac-test --async

# Galaxy S25 / OpenCL
$GC firebase test android run --type game-loop --app encoder-cl.apk \
  --device model=pa1q,version=36,locale=en,orientation=portrait \
  --timeout 600s --project qvac-test --async
```

Each submit prints a **GCS results bucket** like
`gs://test-lab-.../2026-..._XXXX/`. Save it.

---

## 6. Collect results

Firebase has no simple status poll; poll the bucket for the device logcat. When
the run completes, `<bucket>/<model>-<ver>-en-portrait/logcat` appears.

```bash
GC=/home/olya/google-cloud-sdk/bin/gcloud
BASE=gs://test-lab-.../2026-..._XXXX/        # from the submit output
DEV=$($GC storage ls "$BASE" | grep -E '/[a-z0-9]+-[0-9]+-[a-z]+-[a-z]+/$' | head -1)
until $GC storage ls "${DEV}logcat" >/dev/null 2>&1; do sleep 30; done
$GC storage cat "${DEV}logcat" > logcat.txt

grep -a ' ENCBENCH:' logcat.txt | grep -aE 'ENC_RESULT|ENC_COSSIM|ENC_DONE|exit='
```

(`poll_fb.sh <label> <bucket>` in the scratchpad automates the wait+extract.)

**Read the cos, not just the tokens.** A backend can emit 247 tokens of garbage.
`[ENC_COSSIM] ... PASS` (cos ≳0.99) means the GPU output matches the on-device
CPU reference; `FAIL` (cos ~0.6) means the backend is numerically broken.

### Reference results (2026-07-01, q8 mmproj, elephant, iters=10)

| Device | Backend | tokens | avg ms | cos vs CPU | verdict |
|---|---|---|---|---|---|
| Pixel 9 Pro `caiman` | Vulkan (Mali-G715) | 247 | ~4946 | **0.9954** | ✅ correct |
| Galaxy S25 `pa1q` | OpenCL (Adreno 830) | 247 | ~248 | **0.6367** | ❌ runs but WRONG |

Adreno-OpenCL is fast but mis-computes some ViT op → not usable until fixed.
Full context in memory `qwen3vl-standalone-encoder.md`.

---

## Fast iteration summary

- **Kernel edit → verify (desktop):** edit `ggml/src/ggml-vulkan/` or
  `ggml/src/ggml-opencl/` → `cmake --build build-encoder --target
  qwen3vl-encoder` (rebuilds only the backend + relinks the tiny exe) → run
  `--backend all` → check `[ENC_COSSIM]`. Seconds.
- **On-device confirm:** rebuild the android backend, repackage the one APK,
  `--async` submit, poll logcat. ~15–25 min/run (upload + shared device).
