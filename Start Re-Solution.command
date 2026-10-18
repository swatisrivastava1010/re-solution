#!/bin/bash
# Double-click this file to start Re:Solution on a Mac.
# Anything you add after the file name is passed to the app, for example:
#   bash "Start Re-Solution.command" --no-browser --port 7861

cd "$(dirname "$0")" || exit 1

# The uv installer puts uv here. Only this window's PATH changes: the launcher
# never edits your shell settings.
UV_BIN_DIR="$HOME/.local/bin"
export PATH="$UV_BIN_DIR:$PATH"
MIN_CUDA_DRIVER=580 # the cu130 PyTorch build needs NVIDIA driver 580 or newer,
MIN_COMPUTE_CAP=75  # and a card it has code for: compute capability 7.5 (RTX 20 / GTX 16) or newer

pause_then_exit() {
    echo
    echo "$1"
    read -r -p "Press Enter to close."
    exit "$2"
}

# A lone copy of this file can't start anything: the rest of the app must be next to it.
if [ ! -f pyproject.toml ]; then
    pause_then_exit "Sorry, Re:Solution can't start from here: the rest of its folder is missing. If you got it as a zip file, double-click the zip to unpack it, then start Re:Solution from the new folder." 1
fi

# --- Which PyTorch build to use: the same rules as install.sh and install.ps1 ---

# Prints the NVIDIA driver's major version (for example 590), or nothing when there is
# no working NVIDIA driver.
nvidia_driver_major() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local version
    version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)" || return 0
    version="${version%%$'\n'*}" # the first graphics card
    version="${version%%.*}"
    version="${version//[[:space:]]/}"
    case "$version" in '' | *[!0-9]*) return 0 ;; esac
    printf '%s\n' "$version"
}

# Prints the first NVIDIA card's compute capability (its generation, as the graphics
# software counts it) as a whole number: 61 for a GTX 1080's 6.1, 86 for an RTX 3060's 8.6.
# Prints nothing when it can't be read. Drivers older than 510 don't know the question.
nvidia_compute_cap() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || return 0
    cap="${cap%%$'\n'*}" # the first graphics card
    cap="${cap//[[:space:]]/}"
    case "$cap" in [0-9].[0-9] | [0-9][0-9].[0-9]) printf '%s\n' "${cap%.*}${cap#*.}" ;; esac
}

# Prints what this computer's NVIDIA card means for the PyTorch build:
#   none        no NVIDIA driver (so no NVIDIA card to use)
#   no-answer   nvidia-smi is there but didn't answer sensibly this time
#   old-driver  a driver older than $MIN_CUDA_DRIVER
#   old-card    a card the CUDA 13 build has no code for, such as the GTX 10 series:
#               driver 580 is the last one for those cards, so they report it too
#   ok          the cu130 build will work
nvidia_status() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo none && return 0; }
    local major cap
    major="$(nvidia_driver_major)"
    if [ -z "$major" ]; then
        echo no-answer
    elif [ "$major" -lt "$MIN_CUDA_DRIVER" ]; then
        echo old-driver
    else
        cap="$(nvidia_compute_cap)"
        if [ -z "$cap" ]; then
            echo no-answer
        elif [ "$cap" -lt "$MIN_COMPUTE_CAP" ]; then
            echo old-card
        else
            echo ok
        fi
    fi
}

# Prints "cpu" or "cu130".
detect_torch_variant() {
    if [ "$(uname -s)" = "Darwin" ]; then
        echo cpu # Apple GPU support (MPS) is part of the regular Mac build
        return 0
    fi
    if [ "$(nvidia_status)" = ok ]; then
        echo cu130
    else
        echo cpu
    fi
}

# Prints the build saved by the installer (or an earlier start), if it suits this computer.
# Nothing is printed when it should be detected again.
saved_torch_variant() {
    [ -f .torch-variant ] || return 0
    local saved on_mac=0
    saved="$(tr -d '[:space:]' <.torch-variant)" || return 0
    [ "$(uname -s)" = "Darwin" ] && on_mac=1
    case "$saved" in
        # Off a Mac, "cpu" is checked again (that's quick): the folder may have been copied
        # from a Mac, or the NVIDIA driver may have been updated since.
        cpu) if [ "$on_mac" -eq 1 ]; then echo cpu; fi ;;
        # "cu130" is kept while nvidia-smi is there but doesn't answer this time: a driver
        # hiccup shouldn't swap 3 GB of PyTorch. It's checked again when this computer
        # clearly can't use it: no NVIDIA driver at all (for example a folder copied from
        # another PC), or a card or driver too old for it. There's no CUDA build for Macs.
        cu130)
            if [ "$on_mac" -eq 0 ]; then
                case "$(nvidia_status)" in ok | no-answer) echo cu130 ;; esac
            fi
            ;;
    esac
}

if [ "$(uname -s)" = "Darwin" ]; then
    # The app's packages only exist for Apple Silicon on macOS 14 or newer, so say that
    # plainly instead of letting the install fail with a cryptic error. (A Terminal opened
    # "using Rosetta" reports x86_64, so ask the hardware too.)
    if [ "$(uname -m)" != "arm64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]; then
        pause_then_exit "Sorry, Re:Solution needs a Mac with Apple Silicon (M1 or newer)." 1
    fi
    macos_major="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
    case "$macos_major" in '' | *[!0-9]*) macos_major=0 ;; esac
    if [ "$macos_major" -lt 14 ]; then
        pause_then_exit "Sorry, Re:Solution needs macOS 14 (Sonoma) or newer. You can update in System Settings > General > Software Update." 1
    fi
fi

variant="$(saved_torch_variant)"
if [ -z "$variant" ]; then
    variant="$(detect_torch_variant)"
    # Remember it for next time; a read-only folder just means detecting again.
    { printf '%s\n' "$variant" >.torch-variant; } 2>/dev/null
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "First-time setup: installing uv (a small tool that installs everything else)…"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$UV_BIN_DIR" sh
    hash -r
    if ! command -v uv >/dev/null 2>&1; then
        pause_then_exit "Couldn't install uv. Check that this Mac is connected to the internet, then try again." 1
    fi
fi

if [ "$(uname -s)" = "Darwin" ]; then
    echo "Starting Re:Solution. The first start downloads about 400 MB and can take a few minutes."
elif [ "$variant" = "cu130" ]; then
    echo "Starting Re:Solution with your NVIDIA graphics card. The first start downloads about 3 GB and can take 10 minutes or more."
else
    if [ "$(nvidia_status)" = old-card ]; then
        echo "Using the processor (CPU only: slow): this NVIDIA graphics card is too old for the graphics software Re:Solution uses. It needs a GeForce RTX 20 or GTX 16 series card, or newer."
    else
        echo "Using the processor (CPU only: slow): no NVIDIA graphics card with driver $MIN_CUDA_DRIVER or newer was found."
    fi
    echo "Starting Re:Solution. The first start downloads about 400 MB and can take a few minutes."
fi
echo "Keep this window open while you use Re:Solution. Close it to stop."
echo

# --frozen installs exactly the versions in uv.lock; --no-dev skips the test tools;
# --extra picks the PyTorch build.
uv run --frozen --no-dev --extra "$variant" python -m upscaler serve "$@"
status=$?

# 130 means someone pressed Ctrl+C, which is a normal way to stop.
if [ "$status" -ne 0 ] && [ "$status" -ne 130 ]; then
    pause_then_exit "Something went wrong." "$status"
fi
exit 0
