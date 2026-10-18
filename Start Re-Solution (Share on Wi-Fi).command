#!/bin/bash
# Double-click this file to start Re:Solution and let phones and computers on
# your Wi-Fi use it too. Everything else is the same as "Start Re-Solution.command".

echo "Sharing on Wi-Fi: phones and computers on the same Wi-Fi can open the address shown below."
echo "If your Mac asks whether to accept incoming network connections, click Allow."
echo

# Run the main launcher through bash so it works even if a download lost its
# "executable" permission.
exec bash "$(dirname "$0")/Start Re-Solution.command" --share "$@"
