import re
import csv

variables = set()
with open('question_answer_formats.csv') as csv_f:
    reader = csv.DictReader(csv_f)
    for row in reader:
        question = row['questionFormat']
        matches = re.finditer("\[([A-aZ-z|\_|\-]+)\]", question)
        for match in matches:
            variables.add(match.group())

with open('unique_question_variables.csv', 'w', newline='') as csv_output_f:
    to_write = list(sorted(variables))
    writer = csv.writer(csv_output_f)
    writer.writerow(['unqiueVariableName'])
    for string in to_write:
        writer.writerow([string])
