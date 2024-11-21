#!/bin/bash

# Function to find the correct Python binary
find_python() {
    local required_version="3.9"
    local max_version="4.0"
    local python_bins
    python_bins=$(ls /usr/bin | grep -E '^python[0-9.]*$')

    unalias -a # Temporarily disable aliases

    for bin in $python_bins; do
        if command -v "$bin" &>/dev/null; then
            local python_version
            python_version=$($bin -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
            if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) == "$required_version" ]] && \
               [[ $(echo -e "$python_version\n$max_version" | sort -V | head -n1) != "$max_version" ]]; then
                echo "$bin"
                return
            fi
        fi
    done

    echo -e "You must have a Python version higher than 3.9 and lower than 4.0 to launch this project.\n" >&2
    exit 1
}

echo -e "\n \033[1;32m *** Setting up the project *** \033[0m \n"

# Check if find_python exited with an error
if ! PYTHON=$(find_python); then exit 1
fi

echo -e "Using Python binary: $PYTHON\n"

# Display the platform machine
platform_machine=$($PYTHON -c 'import platform; print(platform.machine())')
echo -e "Platform machine: $platform_machine\n"

# Clone the repository in the current directory
git clone https://github.com/nickpadd/EuropeanFootballLeaguePredictor

# Browse the project directory
cd EuropeanFootballLeaguePredictor || exit

# Create a virtual environment
$PYTHON -m venv EFLPvenv

# Activate the virtual environment
source EFLPvenv/bin/activate

# Install dependencies for the project using the pip associated with the virtual environment
EFLPvenv/bin/pip install -r requirements.txt

echo -e "\n \033[1;32m *** The project has been successfully installed. Perform the following README steps. *** \033[0m \n"