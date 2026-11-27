
import csv

total = 0
scam_count = 0
normal_count = 0

with open("dataset/raw/practice_job_scam.csv", "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total += 1

        if row["job_scam"] == "1":
            scam_count += 1
        else:
            normal_count += 1

print("Tổng số bài:", total)
print("Số bài có dấu hiệu lừa đảo:", scam_count)
print("Số bài không có dấu hiệu lừa đảo:", normal_count)
