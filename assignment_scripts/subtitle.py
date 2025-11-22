import subprocess
import os

def add_hardcoded_subtitles(input_video, input_srt, output_video):
    """
    Add hard-coded (burned-in) subtitles to a video using FFmpeg.
    
    Args:
        input_video: Path to input video file
        input_srt: Path to input SRT subtitle file
        output_video: Path to output video file with subtitles
    """
    
    # Check if input files exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video file not found: {input_video}")
    if not os.path.exists(input_srt):
        raise FileNotFoundError(f"SRT file not found: {input_srt}")
    
    # Escape the SRT file path for FFmpeg (important for Windows paths)
    srt_path_escaped = input_srt.replace('\\', '/').replace(':', '\\:')
    
    # FFmpeg command to burn subtitles into video
    # Using subtitles filter with custom styling
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f"subtitles='{srt_path_escaped}':force_style='FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=1,MarginV=20'",
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-c:v', 'libx264',  # Re-encode video with H.264
        '-crf', '18',  # High quality (lower = better quality, 18 is visually lossless)
        '-preset', 'medium',  # Encoding speed preset
        '-y',  # Overwrite output file if it exists
        output_video
    ]
    
    print("Adding hard-coded subtitles to video...")
    print(f"Input video: {input_video}")
    print(f"Input SRT: {input_srt}")
    print(f"Output video: {output_video}")
    print("\nThis may take a few minutes depending on video length...")
    
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n✓ Subtitles successfully added!")
        print(f"Output saved to: {output_video}")
        
    except subprocess.CalledProcessError as e:
        print("\n✗ Error occurred while processing video:")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print("\n✗ FFmpeg not found. Please install FFmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - Mac: brew install ffmpeg")
        print("  - Linux: sudo apt-get install ffmpeg")
        raise


# Main execution
if __name__ == "__main__":
    # Your file paths
    input_video = "SIDxxxxxxx_Asgmt2Opt3.mp4"
    input_srt = "SIDxxxxxxx Asgmt2Opt3.srt"
    output_video = "SIDxxxxxxx_Asgmt2Opt3_output.mp4"
    
    # Add subtitles
    add_hardcoded_subtitles(input_video, input_srt, output_video)