from pathlib import Path
import shutil, requests, os, json
from datetime import datetime

# Create .env file with the following 2 keys in your script directory or remove the .dotenv import 
# and load_dotenv lines and set/use env vars directly
try:
    import dotenv
    dotenv.load_dotenv()
except:
    pass
### Added _SECRET suffix to avoid these from being logged with AIM
AIM_CLIENT_REQUEST_HEADERS = json.loads(os.environ['AIM_CLIENT_REQUEST_HEADERS_SECRET'])
AIM_REMOTE_SERVER_URL = os.environ['AIM_REMOTE_SERVER_URL_SECRET']

def upload_repo():
    server_url = AIM_REMOTE_SERVER_URL
    repo_path = Path('.aim')
    if not repo_path.exists(): raise ValueError(f"AIM repo not found at {repo_path}")
    zip_filepath = repo_path.resolve().parent/'aim_repo.zip'
    if zip_filepath.exists(): zip_filepath.unlink()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_filename = f'aim_repo_{timestamp}.zip'
    shutil.make_archive(str(zip_filepath.with_suffix('')), 'zip', str(repo_path))

    with open(zip_filepath, 'rb') as f:
        response = requests.post(server_url, files={'file': (upload_filename, f)}, headers=AIM_CLIENT_REQUEST_HEADERS, timeout=1200)
        response.raise_for_status()
    print(f'File upload: {upload_filename} success!\nHTTP RESPONSE:\n{response.text}')

    print('repo_path', repo_path, 'zip_filepath', zip_filepath, 'upload_filename', upload_filename, 'zip_filepath.exists()', zip_filepath.exists())
    response_json = response.json()
    return response_json['upload_id'] if response_json['status'] == 'success' else None

if __name__ == '__main__':
    upload_repo()
