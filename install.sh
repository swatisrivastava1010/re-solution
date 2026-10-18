#!/usr/bin/env bash
# Re:Solution installer for Mac (and Linux).
#
# Paste this one line into Terminal and press Enter:
#   curl -fsSL https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.sh | bash
#
# It installs Re:Solution into ~/Applications/Re-Solution, adds a Re-Solution app to
# ~/Applications with an alias on the Desktop, and starts it. Run the same line again to
# update (close Re:Solution first): your settings, keys, downloaded models and pictures are
# kept. It never asks for your password (no admin rights) and never changes your shell
# settings.
#
# Run it with --help to see the options. When piping, pass options after "bash -s --":
#   curl -fsSL .../install.sh | bash -s -- --no-start
#
# The PyTorch build is chosen with the same rules as the Start Re-Solution launchers and
# install.ps1: every Mac gets "cpu" (it includes Apple GPU support); a PC with an NVIDIA
# card of the GeForce RTX 20 / GTX 16 generation or newer (compute capability 7.5) and
# driver 580 or newer gets "cu130"; anything else gets "cpu".
#
# For a fork (or testing), RE_SOLUTION_REPO_URL and RE_SOLUTION_ZIP_URL change where the
# app is downloaded from, and RE_SOLUTION_PORT where it looks for a running Re:Solution.

# Everything is inside main(), called on the last line, so a download cut short by a bad
# connection can't run half a script.
set -euo pipefail

REPO_URL="${RE_SOLUTION_REPO_URL:-https://github.com/swatisrivastava1010/re-solution}"
REPO_BRANCH="main"
ZIP_URL="${RE_SOLUTION_ZIP_URL:-$REPO_URL/archive/refs/heads/$REPO_BRANCH.zip}"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
WINDOWS_COMMAND="irm https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.ps1 | iex"
UV_BIN_DIR="$HOME/.local/bin"
LAUNCHER_NAME="Start Re-Solution.command"
MIN_CUDA_DRIVER=580  # the cu130 PyTorch build needs NVIDIA driver 580 or newer,
MIN_COMPUTE_CAP=75   # and a card it has code for: compute capability 7.5 (RTX 20 / GTX 16) or newer
MIN_MACOS=14         # PyTorch and MLX need macOS 14 Sonoma or newer
APP_PORT="${RE_SOLUTION_PORT:-7860}" # where a running Re:Solution answers
STEPS=5

INSTALL_DIR="$HOME/Applications/Re-Solution"
SOURCE_DIR=""
NO_START=0
NO_SHORTCUTS=0
DRY_RUN=0

OS=""
VARIANT=""
UV=""
TMP_DIR=""
FRIENDLY_EXIT=0

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$'\033[1m' BLUE=$'\033[34m' GREEN=$'\033[32m' YELLOW=$'\033[33m' RED=$'\033[31m' RESET=$'\033[0m'
else
    BOLD="" BLUE="" GREEN="" YELLOW="" RED="" RESET=""
fi

# --- messages --------------------------------------------------------------------------------

step() { printf '\n%s%s%s/%s %s%s\n' "$BOLD" "$BLUE" "$1" "$STEPS" "$2" "$RESET"; }
info() { printf '    %s\n' "$*"; }
ok() { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }

# die "what went wrong." ["what to do" ...]: explain, then stop.
die() {
    FRIENDLY_EXIT=1
    printf '\n%sSorry, %s%s\n' "$RED" "$1" "$RESET" >&2
    shift
    local line
    for line in "$@"; do printf '%s\n' "$line" >&2; done
    exit 1
}

on_exit() {
    local status=$?
    cleanup
    if [ "$status" -ne 0 ] && [ "$FRIENDLY_EXIT" -eq 0 ]; then
        printf '\n%sSorry, something went wrong while installing Re:Solution.%s\n' "$RED" "$RESET" >&2
        printf 'Check that this computer is connected to the internet, then run the same command again.\n' >&2
        printf 'Your settings, keys and pictures are safe.\n' >&2
    fi
}

cleanup() {
    if [ -n "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
        TMP_DIR=""
    fi
}

usage() {
    cat <<'EOF'
Re:Solution installer for Mac (and Linux).

Usage: bash install.sh [options]

  --dir <folder>      install somewhere else (default: ~/Applications/Re-Solution)
  --source <folder>   copy the app from a folder on this computer instead of GitHub
  --no-shortcuts      don't add the app to ~/Applications or the Desktop
                      (the Re-Solution app is still made, inside the install folder)
  --no-start          don't start Re:Solution at the end
  --dry-run           only check this computer and say what would happen
  -h, --help          show this help

Running it again updates Re:Solution in place and keeps your settings, keys, models
and pictures.
EOF
}

# --- choosing the PyTorch build (the same rules as the launchers and install.ps1) ------------

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

# Apple Silicon, even when Terminal was opened "using Rosetta" (then uname says x86_64).
is_apple_silicon() {
    [ "$(uname -m)" = "arm64" ] || [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = "1" ]
}

# --- small helpers -------------------------------------------------------------------------

# A folder as an absolute path without a trailing slash (the folder may not exist yet).
absolute_path() {
    local path="$1"
    case "$path" in
        "~") path="$HOME" ;;
        "~/"*) path="$HOME/${path#"~/"}" ;;
        /*) ;;
        *) path="$PWD/$path" ;;
    esac
    while [ "${#path}" -gt 1 ] && [ "${path%/}" != "$path" ]; do path="${path%/}"; done
    printf '%s\n' "$path"
}

# The real location of a path whose nearest existing parent is resolved (symlinks included).
canonical_path() {
    local path="$1" tail=""
    while [ ! -d "$path" ]; do
        tail="/$(basename "$path")$tail"
        path="$(dirname "$path")"
    done
    printf '%s%s\n' "$(cd "$path" && pwd -P)" "$tail"
}

is_empty_dir() { [ -d "$1" ] && [ -z "$(ls -A "$1" 2>/dev/null)" ]; }
looks_like_the_app() { [ -f "$1/pyproject.toml" ] && [ -d "$1/upscaler" ]; }
have() { command -v "$1" >/dev/null 2>&1; }

# git that works. On a Mac without the developer tools, /usr/bin/git only pops up an offer
# to install them, so the zip download (curl and unzip come with macOS) is used instead.
have_git() {
    have git || return 1
    if [ "$(uname -s)" = "Darwin" ] && [ "$(command -v git)" = "/usr/bin/git" ]; then
        xcode-select -p >/dev/null 2>&1 || return 1
    fi
    # The developer tools can be there and still not work: Xcode's licence not accepted
    # yet, or left broken by a macOS upgrade. git then only complains, so use the zip.
    git --version >/dev/null 2>&1
}

# True when Re:Solution is already running on this computer: it answers on its port the
# way the app itself checks (cli._probe_running). Never through a proxy: the computer's
# proxy settings could send even 127.0.0.1 there.
app_is_running() {
    have curl || return 1
    local answer
    answer="$(curl -s --noproxy '*' --max-time 5 "http://127.0.0.1:$APP_PORT/api/status" 2>/dev/null)" || return 1
    case "$answer" in *'"is_host"'*) return 0 ;; esac
    return 1
}

# 'it'\''s' quoting, for a path inside a shell command. sed rather than ${var//...}: bash
# versions disagree about backslashes in replacements.
shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# Escapes text for an AppleScript "string".
applescript_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# Copies the app's files from $1 into $2. This computer's own files in $2 (the installed
# packages in .venv, .torch-variant, test pictures in samples/) are left alone.
copy_app_files() {
    local from="$1" to="$2"
    mkdir -p "$to" || return 1
    # The top-level entries are picked with find: macOS tar matches an --exclude at any
    # depth, so excluding samples there would also drop the bundled upscaler/web/samples.
    # The --exclude patterns are only for clutter that can be skipped wherever it is.
    # COPYFILE_DISABLE stops macOS tar from carrying Finder metadata along.
    (cd "$from" &&
        find . -mindepth 1 -maxdepth 1 ! -name .git ! -name .venv ! -name .torch-variant \
            ! -name samples ! -name Re-Solution.app -print0 |
        COPYFILE_DISABLE=1 tar -cf - --exclude __pycache__ --exclude .pytest_cache \
            --exclude .ruff_cache --exclude .DS_Store --exclude '*.part' --null -T -) |
        (cd "$to" && tar -xf -) || return 1
    if [ -f "$from/samples/README.md" ]; then
        mkdir -p "$to/samples" && cp "$from/samples/README.md" "$to/samples/README.md" || return 1
    fi
}

# --- the steps -----------------------------------------------------------------------------

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir)
                [ $# -ge 2 ] || die "--dir needs a folder, for example: --dir ~/Re-Solution"
                INSTALL_DIR="$2"
                shift 2
                ;;
            --dir=*) INSTALL_DIR="${1#*=}" && shift ;;
            --source)
                [ $# -ge 2 ] || die "--source needs a folder, for example: --source ~/Downloads/re-solution-main"
                SOURCE_DIR="$2"
                shift 2
                ;;
            --source=*) SOURCE_DIR="${1#*=}" && shift ;;
            --no-start) NO_START=1 && shift ;;
            --no-shortcuts) NO_SHORTCUTS=1 && shift ;;
            --dry-run) DRY_RUN=1 && shift ;;
            -h | --help) usage && exit 0 ;;
            *) die "this installer doesn't know the option \"$1\"." "Run it with --help to see the options." ;;
        esac
    done
    [ -n "$INSTALL_DIR" ] || die "--dir needs a folder, for example: --dir ~/Re-Solution"
    INSTALL_DIR="$(absolute_path "$INSTALL_DIR")"
    if [ -n "$SOURCE_DIR" ]; then
        SOURCE_DIR="$(absolute_path "$SOURCE_DIR")"
        looks_like_the_app "$SOURCE_DIR" || die "\"$SOURCE_DIR\" doesn't look like a Re:Solution folder." \
            "It should contain pyproject.toml and the upscaler folder."
        local source_real target_real
        source_real="$(canonical_path "$SOURCE_DIR")"
        target_real="$(canonical_path "$INSTALL_DIR")"
        case "$target_real/" in
            "$source_real"/*) die "the install folder can't be the source folder or inside it." \
                "Choose another folder with --dir." ;;
        esac
    fi
}

check_computer() {
    step 1 "Checking your computer…"
    OS="$(uname -s)"
    case "$OS" in
        Darwin)
            is_apple_silicon || die "Re:Solution needs a Mac with Apple Silicon (M1 or newer)." \
                "PyTorch, the engine it runs on, no longer makes a version for Intel Macs."
            local version major
            version="$(sw_vers -productVersion 2>/dev/null || true)"
            major="${version%%.*}"
            case "$major" in '' | *[!0-9]*) major=0 ;; esac
            [ "$major" -ge "$MIN_MACOS" ] || die "Re:Solution needs macOS $MIN_MACOS (Sonoma) or newer. This Mac has macOS ${version:-unknown}." \
                "You can update in System Settings > General > Software Update."
            ok "Mac with Apple Silicon, macOS $version"
            ;;
        Linux)
            ok "Linux ($(uname -m))"
            ;;
        *)
            die "this installer is for Mac (and Linux)." "On Windows, paste this into PowerShell instead:" \
                "  $WINDOWS_COMMAND"
            ;;
    esac

    VARIANT="$(detect_torch_variant)"
    if [ "$OS" = "Darwin" ]; then
        ok "Re:Solution will use your Mac's graphics chip"
    elif [ "$VARIANT" = "cu130" ]; then
        ok "NVIDIA graphics card found (driver $(nvidia_driver_major)): Re:Solution will use it"
    else
        case "$(nvidia_status)" in
            old-card)
                warn "Your NVIDIA graphics card is too old for the graphics software Re:Solution uses, so it will use the processor instead (CPU only: slow)."
                info "Re:Solution can use a GeForce RTX 20 or GTX 16 series card, or newer."
                ;;
            old-driver)
                warn "Your NVIDIA driver ($(nvidia_driver_major)) is older than version $MIN_CUDA_DRIVER, so Re:Solution will use the processor instead (CPU only: slow)."
                info "For much faster upscaling, update the driver from https://www.nvidia.com/drivers and run this installer again."
                ;;
            *)
                warn "No NVIDIA graphics card found, so Re:Solution will use the processor (CPU only: slow)."
                ;;
        esac
    fi
    info "PyTorch build: $VARIANT"

    # Updating under a running copy would swap its files while it uses them, and the start
    # at the end would only reopen the old copy.
    if app_is_running; then
        if [ "$DRY_RUN" -eq 1 ]; then
            warn "Re:Solution is open. Close its window before installing for real."
        else
            die "Re:Solution is open, so it can't be installed or updated right now." \
                "Close its window (or press Ctrl+C in it), then run the same command again. Nothing was changed."
        fi
    fi

    # Free space: a few GB for the app's parts, plus 7 GB if Photo-real is used.
    local probe="$INSTALL_DIR" free_gb
    while [ ! -d "$probe" ]; do probe="$(dirname "$probe")"; done
    free_gb="$(df -Pk "$probe" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }' || true)"
    if [ -n "$free_gb" ] && [ "$free_gb" -lt 10 ]; then
        warn "Only $free_gb GB of free space left. Re:Solution needs a few GB, and Photo-real 7 GB more."
    fi
}

get_uv() {
    step 2 "Getting uv, the small tool that installs everything else…"
    export PATH="$UV_BIN_DIR:$PATH"
    if UV="$(command -v uv)"; then
        ok "uv is already installed"
        # Keep a uv this installer put there up to date, so it can read the newest uv.lock.
        if [ "$UV" = "$UV_BIN_DIR/uv" ] && [ "$DRY_RUN" -eq 0 ]; then
            "$UV" self update --quiet </dev/null >/dev/null 2>&1 || true
        fi
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        info "Would install uv into $UV_BIN_DIR"
        return 0
    fi
    info "Installing uv into $UV_BIN_DIR (your shell settings stay as they are)…"
    if ! curl -LsSf "$UV_INSTALL_URL" | env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$UV_BIN_DIR" sh >/dev/null; then
        die "couldn't install uv." "Check that this computer is connected to the internet, then run the same command again."
    fi
    hash -r
    UV="$(command -v uv || true)"
    [ -n "$UV" ] || die "uv was downloaded but can't be found in $UV_BIN_DIR." "Please run the same command again."
    ok "uv installed"
}

get_app() {
    step 3 "Downloading Re:Solution…"
    if [ -e "$INSTALL_DIR" ] && ! is_empty_dir "$INSTALL_DIR" && ! looks_like_the_app "$INSTALL_DIR"; then
        die "the folder \"$INSTALL_DIR\" already has other things in it." \
            "Re:Solution needs a folder of its own. Choose another one with --dir, or empty this one."
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        if [ -n "$SOURCE_DIR" ]; then
            info "Would copy the app from $SOURCE_DIR into $INSTALL_DIR"
        elif looks_like_the_app "$INSTALL_DIR"; then
            info "Would update the copy in $INSTALL_DIR to the latest version"
        else
            info "Would download the app from $REPO_URL into $INSTALL_DIR"
        fi
        return 0
    fi

    mkdir -p "$(dirname "$INSTALL_DIR")"
    local updating=0
    looks_like_the_app "$INSTALL_DIR" && updating=1

    if [ -n "$SOURCE_DIR" ]; then
        copy_app_files "$SOURCE_DIR" "$INSTALL_DIR" || die "couldn't copy the app from \"$SOURCE_DIR\"."
        ok "Copied from $SOURCE_DIR"
    elif [ -d "$INSTALL_DIR/.git" ] && have_git; then
        # fetch + reset (rather than pull) can't get stuck on old local edits.
        if ! git -C "$INSTALL_DIR" fetch --quiet --depth 1 origin "$REPO_BRANCH" </dev/null ||
            ! git -C "$INSTALL_DIR" reset --quiet --hard FETCH_HEAD; then
            die "couldn't download the update." "Check that this computer is connected to the internet, then run the same command again."
        fi
        ok "Updated to the latest version"
    elif have_git && { [ ! -e "$INSTALL_DIR" ] || is_empty_dir "$INSTALL_DIR"; }; then
        git clone --quiet --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR" </dev/null ||
            die "couldn't download Re:Solution." "Check that this computer is connected to the internet, then run the same command again."
        ok "Downloaded into $INSTALL_DIR"
    else
        # No git, or an earlier copy that wasn't made with git: fetch a fresh copy, then
        # copy it over the old one.
        local fresh="$TMP_DIR/app"
        if have_git; then
            git clone --quiet --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$fresh" </dev/null ||
                die "couldn't download Re:Solution." "Check that this computer is connected to the internet, then run the same command again."
        else
            have unzip || die "this computer has neither git nor unzip, which are needed to download Re:Solution." \
                "Please install one of them, then run the same command again."
            curl -fsSL "$ZIP_URL" -o "$TMP_DIR/app.zip" ||
                die "couldn't download Re:Solution." "Check that this computer is connected to the internet, then run the same command again."
            unzip -q "$TMP_DIR/app.zip" -d "$TMP_DIR/unzipped" || die "the download was damaged." "Please run the same command again."
            local folder
            for folder in "$TMP_DIR"/unzipped/*/; do fresh="${folder%/}" && break; done
            looks_like_the_app "$fresh" || die "the download was damaged." "Please run the same command again."
        fi
        copy_app_files "$fresh" "$INSTALL_DIR" || die "couldn't copy the app into \"$INSTALL_DIR\"."
        if [ "$updating" -eq 1 ]; then ok "Updated to the latest version"; else ok "Downloaded into $INSTALL_DIR"; fi
    fi

    # A zip or a copy can lose the "executable" permission.
    chmod +x "$INSTALL_DIR"/*.command "$INSTALL_DIR/install.sh" 2>/dev/null || true
    printf '%s\n' "$VARIANT" >"$INSTALL_DIR/.torch-variant"
}

install_parts() {
    local size="about 400 MB"
    [ "$VARIANT" = "cu130" ] && size="about 3 GB"
    step 4 "Installing the app's parts (the first time, this downloads $size and can take a while)…"
    if [ "$DRY_RUN" -eq 1 ]; then
        info "Would run: uv sync --frozen --no-dev --extra $VARIANT"
        return 0
    fi
    # Only this script's environment changes: an activated virtual environment must not
    # redirect the install.
    if ! env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT \
        "$UV" sync --frozen --no-dev --extra "$VARIANT" --directory "$INSTALL_DIR" </dev/null; then
        die "couldn't install the app's parts." \
            "Check the internet connection and that there is free disk space, then run the same command again." \
            "If Re:Solution is open, close its window first."
    fi
    ok "Everything is installed"
}

# Builds a tiny AppleScript app that opens Terminal and runs the launcher there.
build_mac_app() {
    local app="$1" command errors
    command="cd $(shell_quote "$INSTALL_DIR") && ./Start\\ Re-Solution.command"
    rm -rf "$app"
    # osacompile chats on stderr even when it works (about signing), so only show that
    # when it fails.
    if ! errors="$(osacompile -o "$app" \
        -e '-- Opens Terminal and starts Re:Solution. Made by install.sh; run the installer again to rebuild it.' \
        -e "set launchCommand to \"$(applescript_escape "$command")\"" \
        -e 'if application "Terminal" is running then' \
        -e '	tell application "Terminal" to do script launchCommand' \
        -e 'else' \
        -e '	-- A Terminal that has just started opens an empty window: use that one.' \
        -e '	tell application "Terminal" to activate' \
        -e '	repeat 50 times' \
        -e '		if (count of windows of application "Terminal") > 0 then exit repeat' \
        -e '		delay 0.1' \
        -e '	end repeat' \
        -e '	if (count of windows of application "Terminal") > 0 then' \
        -e '		tell application "Terminal" to do script launchCommand in window 1' \
        -e '	else' \
        -e '		tell application "Terminal" to do script launchCommand' \
        -e '	end if' \
        -e 'end if' \
        -e 'tell application "Terminal" to activate' \
        2>&1 >/dev/null)"; then
        printf '%s\n' "$errors" >&2
        return 1
    fi
}

make_shortcuts() {
    step 5 "Creating shortcuts…"
    if [ "$OS" != "Darwin" ]; then
        info "Shortcuts are only made on a Mac. To start Re:Solution, run:"
        info "  \"$INSTALL_DIR/$LAUNCHER_NAME\""
        return 0
    fi
    local apps_dir="$HOME/Applications"
    [ "$NO_SHORTCUTS" -eq 1 ] && apps_dir="$INSTALL_DIR"
    local app="$apps_dir/Re-Solution.app" alias="$HOME/Desktop/Re-Solution"
    if [ "$DRY_RUN" -eq 1 ]; then
        info "Would create $app"
        [ "$NO_SHORTCUTS" -eq 1 ] || info "Would add an alias to it on the Desktop"
        return 0
    fi

    if ! mkdir -p "$apps_dir" || ! build_mac_app "$app"; then
        warn "Couldn't create the Re-Solution app. You can still start Re:Solution by double-clicking"
        info "\"$LAUNCHER_NAME\" in $INSTALL_DIR"
        return 0
    fi
    ok "Made the Re-Solution app in $apps_dir"
    if [ "$NO_SHORTCUTS" -eq 1 ]; then
        info "(Nothing was added to ~/Applications or the Desktop.)"
        return 0
    fi

    if [ ! -d "$HOME/Desktop" ]; then
        return 0
    elif [ -e "$alias" ] && [ ! -L "$alias" ]; then
        warn "There's already something called Re-Solution on your Desktop, so no alias was added."
    elif ln -sfn "$app" "$alias" 2>/dev/null; then
        ok "Added Re-Solution to your Desktop"
    else
        warn "Couldn't add Re-Solution to the Desktop (macOS may have asked and been told no). The app is in $apps_dir."
    fi
}

finish() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '\n%sDry run finished: nothing was changed.%s\n' "$BOLD" "$RESET"
        return 0
    fi
    printf '\n%s%sRe:Solution is installed in %s%s\n' "$BOLD" "$GREEN" "$INSTALL_DIR" "$RESET"
    info "To update later, run the same command again. Your settings, keys, models and pictures are kept."
    if [ "$NO_START" -eq 1 ]; then
        if [ "$OS" = "Darwin" ] && [ "$NO_SHORTCUTS" -eq 0 ]; then
            info "Start it any time with the Re-Solution app in your Applications folder or on the Desktop."
        else
            info "Start it any time by opening \"$LAUNCHER_NAME\" in $INSTALL_DIR"
        fi
        return 0
    fi
    if [ "$OS" = "Darwin" ] && [ "$NO_SHORTCUTS" -eq 0 ]; then
        info "Next time, start it with the Re-Solution app in your Applications folder or on the Desktop."
        info "The first time, your Mac may ask whether Re-Solution may control Terminal: click OK."
    fi
    printf '\n%sStarting Re:Solution…%s\n\n' "$BOLD" "$RESET"
    cleanup
    trap - EXIT
    # This window becomes Re:Solution's window. When piped from curl, the keyboard is
    # /dev/tty rather than the pipe.
    if { : </dev/tty; } 2>/dev/null; then
        exec bash "$INSTALL_DIR/$LAUNCHER_NAME" </dev/tty
    fi
    exec bash "$INSTALL_DIR/$LAUNCHER_NAME"
}

main() {
    trap on_exit EXIT
    parse_args "$@"
    printf '%sInstalling Re:Solution%s, the free picture and video upscaler.\n' "$BOLD" "$RESET"
    [ "$DRY_RUN" -eq 1 ] && printf 'Dry run: checking only, nothing will be changed.\n'
    check_computer
    if [ "$DRY_RUN" -eq 0 ]; then
        TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/re-solution-install.XXXXXX")"
    fi
    get_uv
    get_app
    install_parts
    make_shortcuts
    finish
}

main "$@"
