import csv
import sys
import json
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

title = 'Chuyên Viên Phát Triển Khách Hàng Cá Nhân - MRBT'
desc = ''
with open('job13_details.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        desc += f"[{r['title']}]\n{r['content']}\n\n"

req_data = json.dumps({'title': title, 'description': desc}).encode('utf-8')
req = urllib.request.Request(
    'http://localhost:8000/predict',
    data=req_data,
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(req) as resp:
    print('HTTP Status:', resp.status)
    data = json.loads(resp.read().decode('utf-8'))
    print('Title:', data['title'])
    print('Results:')
    for k, v in data['results'].items():
        print(f"  {k:20s}: pred={v['prediction']}, prob={v['probability']}")
