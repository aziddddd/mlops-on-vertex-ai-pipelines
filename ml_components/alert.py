from typing import NamedTuple
from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Input,
    InputPath, 
    Model, 
    Output,
    OutputPath,
    Metrics,
    HTML,
    component
)

def push_slack_notification(
    job_id: str,
    src_run_dt: str,
    text: str,
    channel: str,
    webhook_config_str: str,
    runner: str,
) -> NamedTuple(
    "Outputs",
    [
        ("rtn_code", int),
    ],
):
    def get_secret(
        project_uid: str,
        secret_id: str,
        version_id: str,
    ):
        import google.auth
        from google.oauth2 import service_account
        from google.cloud import secretmanager
        import json

        # Create the Secret Manager client.
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version.
        name = f"projects/{project_uid}/secrets/{secret_id}/versions/{version_id}"

        # Access the secret version.
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")

        return payload

    import pandas as pd
    import requests
    import json

    webhook_config = json.loads(webhook_config_str)
    url = get_secret(**webhook_config)

    details = {
        'job_id': job_id,
        'src_run_dt': src_run_dt,
        'runner': runner,
        'pipeline_status': text,
    }

    df = pd.DataFrame(details.items(), columns=['Details', 'Value'])

    keys_ = list(df.keys())
    assert (
        0 < len(keys_) <= 2
    ), 'dataframe should have atleast 1 column, and maximum of 2 columns'

    json_data = {}
    assert '#' in channel, "Channel name should start with '#'"
    json_data['channel'] = channel

    json_data['blocks'] = []

    table_fields = []
    for key in keys_:
        temp = {}
        temp['type'] = 'mrkdwn'
        temp['text'] = '*{}*'.format(key)
        table_fields.append(temp)

    for key in keys_:
        temp = {}
        temp['type'] = 'plain_text'
        temp['text'] = '\n'.join([str(x) for x in df[key].to_list()])
        table_fields.append(temp)

    json_data['blocks'].append({'type': 'section', 'fields': table_fields})

    if username: json_data['username'] = username;
    if icon_emoji: json_data['icon_emoji'] = icon_emoji;

    rtn_code = requests.post(url, json = json_data)
    print(f"return code : {rtn_code}")
    return (
        rtn_code.status_code,
    )