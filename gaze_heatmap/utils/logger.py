"""
Structured logging utilities.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class SessionLogger:
    """Logger for recording session events with timestamps."""
    
    def __init__(self, session_id: str, output_dir: Optional[Path] = None):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.events = []
        self.output_dir = output_dir
        self.logger = get_logger(f"session.{session_id}")
        
    def log_event(self, event_type: str, data: dict = None):
        """Log a session event."""
        timestamp = (datetime.now() - self.start_time).total_seconds()
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'data': data or {}
        }
        self.events.append(event)
        self.logger.info(f"{event_type}: {data}")
        
    def log_calibration_point(self, point_idx: int, screen_pos: tuple, 
                               gaze_data: dict):
        """Log a calibration point recording."""
        self.log_event('calibration_point', {
            'point_idx': point_idx,
            'screen_pos': screen_pos,
            'gaze_data': gaze_data
        })
        
    def log_gaze_sample(self, screen_pos: tuple, raw_gaze: tuple, 
                        confidence: float):
        """Log a gaze sample."""
        self.log_event('gaze_sample', {
            'screen_pos': screen_pos,
            'raw_gaze': raw_gaze,
            'confidence': confidence
        })
        
    def log_fixation(self, center: tuple, duration_ms: float):
        """Log a detected fixation."""
        self.log_event('fixation', {
            'center': center,
            'duration_ms': duration_ms
        })
        
    def save(self, path: Optional[Path] = None):
        """Save session log to JSON file."""
        import json
        
        if path is None:
            if self.output_dir:
                path = self.output_dir / f"{self.session_id}_events.json"
            else:
                return
                
        with open(path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat(),
                'events': self.events
            }, f, indent=2)

