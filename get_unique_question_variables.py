import re
import csv
import argparse

def run(input_file, output_file, column):
    variables = set()
    with open(input_file) as csv_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            question = row[column]
            matches = re.finditer("\[([A-aZ-z|\_|\-]+)\]", question)
            for match in matches:
                variables.add(match.group())

    with open(output_file, 'w', newline='') as csv_output_f:
        to_write = list(sorted(variables))
        writer = csv.writer(csv_output_f)
        writer.writerow(['unqiueVariableName'])
        for string in to_write:
            writer.writerow([string])

def main():
    parser = argparse.ArgumentParser(description="Extract all the unique variable names from a csv file (variables have the form [VAR])")
    parser.add_argument('-in_file', type=str, help='the input csv file')
    parser.add_argument('-out_file', type=str, help='the output csv file')
    parser.add_argument('-column', type=str, help="the column in the csv file to search")

    args = parser.parse_args()
    run(args.in_file, args.out_file, args.column)

if __name__ == "__main__":
    main()
