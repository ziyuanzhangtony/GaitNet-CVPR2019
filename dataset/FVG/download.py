# Temporal download page:
# https://drive.google.com/drive/folders/10e3rhSkOPyAzVFVlskm1n2iadroQu7G-




# import io
# import pickle
# import os.path
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.http import MediaIoBaseDownload
# SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
# creds = None
# if os.path.exists('token.pickle'):
#     with open('token.pickle', 'rb') as token:
#         creds = pickle.load(token)
# # If there are no (valid) credentials available, let the user log in.
# if not creds or not creds.valid:
#     if creds and creds.expired and creds.refresh_token:
#         creds.refresh(Request())
#     else:
#         flow = InstalledAppFlow.from_client_secrets_file(
#             'credentials.json', SCOPES)
#         creds = flow.run_local_server()
#     # Save the credentials for the next run
#     with open('token.pickle', 'wb') as token:
#         pickle.dump(creds, token)
# drive_service = build('drive', 'v3', credentials=creds)
# # Call the Drive v3 API
# results = drive_service.files().list(
#     pageSize=10, fields="nextPageToken, files(id, name)").execute()
# items = results.get('files', [])
#
# if not items:
#     print('No files found.')
# else:
#     print('Files:')
#     for item in items:
#         print(u'{0} ({1})'.format(item['name'], item['id']))
#
#
# file_id = '1hzE5rNK1e6hrERyxMlm1vcE0GuKJwFxo'
# request = drive_service.files().get_media(fileId=file_id)
# fh = io.BytesIO()
# downloader = MediaIoBaseDownload(fh, request)
# done = False
# while done is False:
#     status, done = downloader.next_chunk()
#     print("Download %d%%." % int(status.progress() * 100))