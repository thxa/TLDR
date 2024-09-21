import uvicorn
from .rest_api import app
import sys

def main():
    config = uvicorn.Config("vehicle_routing:app",
                            port=8080,
                            log_config=sys.path[-1]+"/../"+"logging.conf",
                            use_colors=True)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
