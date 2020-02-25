'''
Title: Generate Input Data
Author: Chidi Ewenike
Date: 01/24/2020
Organization: Cal Poly CSAI
Description: Generates input text data based on a list of prefixes and suffixes.
'''

class Gen_Data:
    def __init__(self):
        self.prefixes_1 = ["Can you give me", "Can you provide me", "Please give me", "Give me", "Tell me", "Reveal", "I would like", "Just tell me", "It would be nice if you gave me", "I would like to know", "Where can I find", "How do I find", "Provide me", "I would prefer", "I would appreciate", "It would be nice to have", "How can I find", "Can you share with me", "Share with me", "I need"] 

        self.suffix_1 = ["0[PROF]'s office hours", "1the units for [COURSE]", "0the office hours of [PROF]", "1[COURSE]'s units", "2[PROF]'s email", "2[PROF]'s contact information", "1the number of units for [COURSE]", "2a way to contact [PROF]",  "6information when [COURSE] is available", "6availability of [COURSE]", "6when [COURSE] is offered", "7[CLUB]'s contact information", "7how to reach [CLUB]", "7a way to contact [CLUB]", "8where [CLUB] meets", "8the location of [CLUB]'s meetings", "8where to find [CLUB]"]
        # "0the office hours of [PROF]", "3information about [COURSE]", , "4information about [PROF]", "4[PROF]'s expertise", "4the expertise of [PROF]", "4what [PROF] specializes in", "4the specialty of [PROF]", "5[BUILDING] is located", "5the location of [BUILDING]", "5where [BUILDING] is", "5where I can find [BUILDING]",
        self.reference_questions = [
            "When are [PROF]'s office hours?",
            "How many units is [COURSE]?",
            "When are[PROF]'s office hours?",
            "How many units is [COURSE]?",
            "What is [PROF]'s email?",
            "What is [PROF]'s contact information?",
            "How many units is [COURSE]?",
            "How can I contact [PROF]?",
            "When is [COURSE] available?",
            "When is [COURSE] available?",
            "When is [COURSE] offered?",
            "What is the general contact information for [CLUB]?",
            "Who is the chief contact for [CLUB]?",
            "Where is [CLUB]?",
            "What is the location of [CLUB]?",
            "Where is [CLUB]?",
            "Where is [CLUB]?"
        ]

        self.labels = [
            "The office hours for [PROF] are [OFFICE HOURS].", # 0
            "[COURSE] is [UNITS].", # 1
            "[PROF]'s email is [EMAIL].", # 2
            "[COURSE] is about [INFORMATION].", # 3
            "[PROF] is in the [MAJOR] department and teaches [TEACHES].", # 4
            "[BUILDING] is located at [LOCATION].", # 5
            "[COURSE] is offered [QUARTERS].", # 6
            "[CLUB] contact information is [CONTACT INFORMATION].", # 7
            "[CLUB] meets [MEETING].", # 8
            ]

        self.gen_data = []
        self.gen_labels = []

    def generate(self):
        print("len(suffixes): {} len(reference_questions): {}".format(len(self.suffix_1), len(self.reference_questions)))
        for i in range(len(self.suffix_1)):
            for j in range(len(self.prefixes_1)):
                paraphrase = self.prefixes_1[j] + " " + self.suffix_1[i][1:]
                self.gen_data.append((self.reference_questions[i], paraphrase))
                self.gen_labels.append(self.labels[int(self.suffix_1[i][0])])
   
        return self.gen_data, self.gen_labels


if __name__ == '__main__':
    data = Gen_Data()
    print(data.generate())
