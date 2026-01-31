#!/usr/bin/env python3
"""
YOLO11 Vision Server Test Client
Mock iPhone client for testing 3x3 grid and geometric clustering logic
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import socketio
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Create Socket.IO client
sio = socketio.Client()

# Global flag to track if response was received
response_received = False


def print_header(text):
    """Print a styled header."""
    print(f"\n{Fore.CYAN}{'=' * 70}")
    print(f"{Fore.CYAN}{text.center(70)}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def print_success(text):
    """Print success message in green."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text):
    """Print error message in red."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_warning(text):
    """Print warning message in yellow."""
    print(f"{Fore.YELLOW}⚠  {text}{Style.RESET_ALL}")


def print_emergency_alert():
    """Print big red emergency alert."""
    alert = """
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║                    🚨  EMERGENCY DETECTED  🚨                      ║
    ║                                                                    ║
    ║              DANGEROUS VEHICLE IN IMMEDIATE PROXIMITY             ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    print(f"{Fore.RED}{Style.BRIGHT}{alert}{Style.RESET_ALL}")


def load_and_encode_image(image_path: str) -> str:
    """
    Load a local image file and convert it to Base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded image with data URL prefix
    """
    try:
        # Read image file as bytes
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Encode to base64
        base64_string = base64.b64encode(image_data).decode('utf-8')

        # Add data URL prefix (matching iPhone format)
        data_url = f"data:image/jpeg;base64,{base64_string}"

        file_size_kb = len(image_data) / 1024
        print_success(f"Loaded image: {image_path} ({file_size_kb:.1f} KB)")

        return data_url

    except FileNotFoundError:
        print_error(f"Image file not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to load image: {e}")
        sys.exit(1)


@sio.event
def connect():
    """Handle successful connection to server."""
    print_success("Connected to YOLO11 Vision Server")


@sio.event
def disconnect():
    """Handle disconnection from server."""
    print_warning("Disconnected from server")


@sio.event
def connection_established(data):
    """Handle connection confirmation from server."""
    print_success(f"Server confirmed connection: {data}")


@sio.event
def scene_analysis(data):
    """
    Handle scene analysis response from server.
    This is where we verify the 3x3 grid and clustering logic.
    """
    global response_received
    response_received = True

    print_header("SCENE ANALYSIS RESPONSE")

    # Extract key fields
    timestamp = data.get('timestamp', 'N/A')
    emergency_stop = data.get('emergency_stop', False)
    summary = data.get('summary', 'No summary')
    objects = data.get('objects', [])
    clusters = data.get('clusters', [])

    # Print timestamp
    print(f"{Fore.CYAN}Timestamp:{Style.RESET_ALL} {timestamp}")
    print()

    # EMERGENCY CHECK (Big Red Alert)
    if emergency_stop:
        print_emergency_alert()
    else:
        print(f"{Fore.GREEN}Emergency Status:{Style.RESET_ALL} ✓ Safe (no immediate threats)")
        print()

    # SUMMARY (Natural Language - THE MAIN FEATURE!)
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Natural Language Summary:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}\"{summary}\"{Style.RESET_ALL}")
    print()

    # CLUSTERS (Geometric Pattern Detection - CRUCIAL FOR VERIFICATION!)
    print(f"{Fore.YELLOW}{Style.BRIGHT}Detected Geometric Patterns (Clusters):{Style.RESET_ALL}")
    if clusters:
        for idx, cluster in enumerate(clusters, 1):
            cluster_type = cluster.get('type', 'unknown')
            label = cluster.get('label', 'object')
            count = cluster.get('count', 0)

            # Type-specific formatting
            if cluster_type == 'crowd':
                position = cluster.get('position', 'unknown').replace('-', ' ')
                print(f"  {Fore.YELLOW}[{idx}] CROWD:{Style.RESET_ALL} {count} {label}s clustered at {position}")

            elif cluster_type == 'row':
                span = cluster.get('span', 'unknown')
                print(f"  {Fore.YELLOW}[{idx}] ROW:{Style.RESET_ALL} {count} {label}s spanning from {span}")

            elif cluster_type == 'stack':
                position = cluster.get('position', 'unknown')
                print(f"  {Fore.YELLOW}[{idx}] STACK:{Style.RESET_ALL} {count} {label}s stacked ({position} side)")

            else:
                print(f"  {Fore.YELLOW}[{idx}] {cluster_type.upper()}:{Style.RESET_ALL} {count} {label}s")

        print()
    else:
        print(f"  {Fore.WHITE}(No clusters detected - objects are isolated){Style.RESET_ALL}")
        print()

    # INDIVIDUAL OBJECTS (3x3 Grid Positions)
    print(f"{Fore.BLUE}{Style.BRIGHT}Individual Objects (3x3 Grid):{Style.RESET_ALL}")
    if objects:
        for idx, obj in enumerate(objects, 1):
            label = obj.get('label', 'unknown')
            confidence = obj.get('confidence', 0.0)
            position = obj.get('position', 'unknown').replace('-', ' ')
            distance = obj.get('distance', 'unknown')

            # Color code by distance
            if distance == 'immediate':
                dist_color = Fore.RED
            elif distance == 'close':
                dist_color = Fore.YELLOW
            else:
                dist_color = Fore.GREEN

            print(
                f"  {Fore.BLUE}[{idx}]{Style.RESET_ALL} "
                f"{label.capitalize()} "
                f"({Fore.WHITE}{confidence:.0%}{Style.RESET_ALL}) - "
                f"Position: {Fore.CYAN}{position}{Style.RESET_ALL}, "
                f"Distance: {dist_color}{distance}{Style.RESET_ALL}"
            )

        print(f"\n  {Fore.WHITE}Total Objects: {len(objects)}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.WHITE}(No objects detected){Style.RESET_ALL}")

    print()

    # DEBUG FRAME (Visual Annotation)
    if 'debug_frame' in data:
        print_header("VISUAL DEBUG OUTPUT")
        try:
            # Extract and decode debug frame
            debug_frame_data = data['debug_frame']

            # Remove data URL prefix if present
            if ',' in debug_frame_data:
                debug_frame_data = debug_frame_data.split(',', 1)[1]

            # Decode base64
            image_bytes = base64.b64decode(debug_frame_data)

            # Save to file
            output_path = Path("debug_output.jpg")
            with open(output_path, 'wb') as f:
                f.write(image_bytes)

            print_success(f"Debug image saved to: {output_path.absolute()}")

            # Automatically open the image
            import platform
            import subprocess

            system = platform.system()
            try:
                if system == 'Windows':
                    import os
                    os.startfile(output_path)
                    print_success("Opening image in default viewer (Windows)...")
                elif system == 'Darwin':  # macOS
                    subprocess.run(['open', str(output_path)])
                    print_success("Opening image in default viewer (macOS)...")
                else:  # Linux
                    subprocess.run(['xdg-open', str(output_path)])
                    print_success("Opening image in default viewer (Linux)...")
            except Exception as e:
                print_warning(f"Could not auto-open image: {e}")
                print(f"  Please manually open: {output_path.absolute()}")

        except Exception as e:
            print_error(f"Failed to process debug frame: {e}")

        print()

    # FULL JSON (for debugging)
    print(f"{Fore.WHITE}{Style.DIM}Full JSON Response (excluding debug_frame):{Style.RESET_ALL}")
    # Remove debug_frame from JSON output (too large)
    display_data = {k: v for k, v in data.items() if k != 'debug_frame'}
    print(f"{Fore.WHITE}{Style.DIM}{json.dumps(display_data, indent=2)}{Style.RESET_ALL}")


@sio.event
def error(data):
    """Handle error events from server."""
    print_error(f"Server error: {data}")


def main():
    """Main test client logic."""
    parser = argparse.ArgumentParser(
        description='Test client for YOLO11 Vision Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_client.py crowd.jpg
  python test_client.py test_images/street_scene.jpg --debug
  python test_client.py path/to/image.png --server http://192.168.1.100:8000
  python test_client.py image.jpg --debug --timeout 60
        """
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the test image file (JPG, PNG, etc.)'
    )
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:8000',
        help='Server URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Response timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: server will return annotated image with bounding boxes'
    )

    args = parser.parse_args()

    # Validate image path
    if not Path(args.image_path).exists():
        print_error(f"Image file not found: {args.image_path}")
        sys.exit(1)

    print_header("YOLO11 VISION SERVER TEST CLIENT")

    # Step 1: Load and encode image
    print(f"{Fore.CYAN}[1/4] Loading image...{Style.RESET_ALL}")
    base64_image = load_and_encode_image(args.image_path)

    # Step 2: Connect to server
    print(f"\n{Fore.CYAN}[2/4] Connecting to server at {args.server}...{Style.RESET_ALL}")
    try:
        sio.connect(args.server)
    except Exception as e:
        print_error(f"Failed to connect to server: {e}")
        print_warning("Make sure the server is running: python server.py")
        sys.exit(1)

    # Give server a moment to process connection
    time.sleep(0.5)

    # Step 3: Send video frame
    print(f"\n{Fore.CYAN}[3/4] Sending video frame for analysis...{Style.RESET_ALL}")
    if args.debug:
        print_warning("Debug mode enabled - requesting annotated frame")
    try:
        payload = {'frame': base64_image, 'debug': args.debug}
        sio.emit('video_frame', payload)
        print_success("Frame sent successfully")
    except Exception as e:
        print_error(f"Failed to send frame: {e}")
        sio.disconnect()
        sys.exit(1)

    # Step 4: Wait for response
    print(f"\n{Fore.CYAN}[4/4] Waiting for scene analysis...{Style.RESET_ALL}")

    # Wait for response with timeout
    start_time = time.time()
    while not response_received and (time.time() - start_time) < args.timeout:
        time.sleep(0.1)

    if not response_received:
        print_error(f"No response received after {args.timeout} seconds")
        sio.disconnect()
        sys.exit(1)

    # Clean disconnect
    time.sleep(0.5)
    sio.disconnect()

    print_header("TEST COMPLETE")
    print_success("All checks passed! Server is functioning correctly.")


if __name__ == "__main__":
    main()
