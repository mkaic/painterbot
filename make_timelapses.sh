
ffmpeg -y -framerate 24 -pattern_type "glob" -i "painting_timelapse_frames/*.jpg" -c:v libx264 -pix_fmt yuv420p painting_timelapse.mp4
ffmpeg -y -framerate 24 -pattern_type "glob" -i "optimization_timelapse_frames/*.jpg" -c:v libx264 -pix_fmt yuv420p optimization_timelapse.mp4