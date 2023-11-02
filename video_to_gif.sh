INPUT_FILENAME=$1
OUTPUT_FILENAME=$2
FPS=12
ffmpeg -y \
    -i $INPUT_FILENAME \
    -vf "fps=$FPS,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 \
     $OUTPUT_FILENAME