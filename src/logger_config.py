import datetime
import logging
import os


def setup_logger():
    # Get current timestamp for unique log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/app_{timestamp}.log"

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="w",  # 'w' creates a new file each time
    )

    # Add console handler to see logs in terminal too
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)  # Only show INFO and above in console
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # Let user know where logs are being saved
    print(f"Logs are being saved to: {os.path.abspath(log_filename)}")

    return log_filename
