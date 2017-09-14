# Copyright 2017, Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from gcs_helper import archive_and_upload

credentials = GoogleCredentials.get_application_default()

BODY = {
    'source': {
        'storageSource': {
            'bucket': None,
            'object': None
        }
    },
    'steps': [
        {
            'name': 'gcr.io/cloud-builders/docker',
            'args': [
                'build',
                '-t',
                'gcr.io/$PROJECT_ID/hpsearch',
                '/workspace'
            ]
        }
    ],
    'images': [
        'gcr.io/$PROJECT_ID/hpsearch'
    ]
}

def build(project_id, source_dir, bucket_name):
	"""This uses the provided bucket (must be writable by the project)
	to store intermediate source archive, and then builds an image.

	"""
	archive_and_upload(bucket_name, source_dir)

	body = BODY
	body['source']['storageSource']['bucket'] = bucket_name
	body['source']['storageSource']['object'] = '{}.zip'.format(source_dir)

	service = discovery.build('cloudbuild', 'v1', credentials=credentials)
	build = service.projects().builds().create(projectId=project_id, body=body).execute()

	return build

