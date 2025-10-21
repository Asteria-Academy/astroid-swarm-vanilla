import logging
import os
import time
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class LogflareLogger:
    def __init__(self):
        self.api_key = os.getenv('LOGFLARE_API_KEY')
        self.endpoint = os.getenv('LOGFLARE_ENDPOINT')
        
        if not all([self.api_key, self.endpoint]):
            logger.error("Missing required Logflare environment variables")
            raise ValueError("Missing required Logflare environment variables")
            
        logger.info(f"LogflareLogger initialized with endpoint: {self.endpoint}")
        
    def log_event(self, event_type, data):
        """Log an event to Logflare"""
        try:
            # Add timestamp
            data['timestamp'] = time.time()
            
            # Prepare the log message
            log_message = {
                'event_type': event_type,
                'data': data,
                'timestamp': time.time()
            }
            
            # Send to Logflare
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Format the log message according to Logflare's requirements
            formatted_log = {
                'event_message': f'{event_type}: {json.dumps(data)}',
                'metadata': {
                    'timestamp': data.get('timestamp', time.time()),
                    **data
                }
            }
            
            response = requests.post(
                f'https://api.logflare.app/logs?source={self.endpoint}',
                headers=headers,
                json=formatted_log
            )
            
            response.raise_for_status()
            logger.info("Successfully logged event to Logflare")
            
        except Exception as e:
            logger.error(f"Failed to log event: {str(e)}")
            raise

    @staticmethod
    def setup_logflare_handler():
        """Set up a logging handler for Logflare"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the root logger
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

# Create a global instance
logflare_logger = LogflareLogger()

def setup_logflare_handler():
    """Set up a logging handler for Logflare"""
    class LogflareHandler(logging.Handler):
        def emit(self, record):
            try:
                # Skip logging if it's a Logflare log to prevent recursion
                if record.name == __name__:
                    return
                
                log_data = {
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'name': record.name,
                    'timestamp': record.created,
                    'extra_data': getattr(record, 'extra_data', {})
                }
                logflare_logger.log_event('application_log', log_data)
            except Exception as e:
                print(f"Error in LogflareHandler: {str(e)}")
    
    handler = LogflareHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(handler)

# Set up the handler when this module is imported
setup_logflare_handler()
