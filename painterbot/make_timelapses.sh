
ffmpeg -y -hide_banner -loglevel error -framerate 24 -pattern_type "glob" -i "timelapse_frames_optimization/*.jpg" -c:v libx264 -pix_fmt yuv420p timelapse_optimization.mp4
ffmpeg -y -hide_banner -loglevel error -framerate 24 -pattern_type "glob" -i "timelapse_frames_painting/*.jpg" -c:v libx264 -pix_fmt yuv420p timelapse_painting.mp4