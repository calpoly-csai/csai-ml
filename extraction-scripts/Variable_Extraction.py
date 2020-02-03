"""
Title: Variable Extraction Class
Author: Chidi Ewenike
Date: 12/15/2019
Organization: Cal Poly CSAI
Description: Applies variable extraction to input queries.

"""

import spacy
from spacy import displacy

from spacy.matcher import Matcher
from spacy.tokens import Span 

import warnings 
import gensim
import json
import time 
import numpy as np
import csv
import sys

from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

#from gen_data import Gen_Data 

class Variable_Extraction:
    def __init__(self):
        self.titles = {"professor", "doctor", "dr", "prof", "mr", "mister", "mrs", "miss", "ms", "instructor"}  
        self.variable = ""
        self.string = ""
        self.poi = ""
        self.database = {}

    def add_cross_name(self, temp_database, word_object):
        if word_object.name not in self.database:
            self.database[word_object.name] = word_object
            self.database[word_object.name.lower()] = word_object

        if not(isinstance(word_object, Professor)):
            for word in word_object.crosslisted:
                if word not in self.database:
                    self.database[word.lower()] = word_object
                    self.database[word] = word_object
    def extract_variables(self, sent):
        
        prefix = ""
        suffix = ""
        prv_tok_dep = ""
        prv_tok_text = ""
        
        last_found_var = ""

        poss_bool = False

        nlp = spacy.load('en_core_web_sm')
        
        print(self.database)
        for tok in nlp(sent):
            print("\nCurr Word: ", tok.text)
            print("Dep: ", tok.dep_)
            print("Prefix: ", prefix)
            print("prv_tok_dep: ", prv_tok_dep)
            print("prv_tok_text: ", prv_tok_text)
            print("poi: ", self.poi)
            print("variable: ", self.variable)
            print("LFV: ", last_found_var)
            print()
            if tok.dep_ != "punct":
                
                temp_phrase = prefix + " " + tok.text

                if temp_phrase in self.database and prv_tok_dep == "compound":
                    last_found_var = temp_phrase

                elif temp_phrase.replace(" ", "") in self.database:
                    last_found_var = temp_phrase.replace(" ", "")
 
                if tok.dep_ == "compound":

                    if prv_tok_dep == "compound" and tok.text.lower() not in self.titles:
                        prefix = temp_phrase

                    else:
                        prefix = tok.text.replace(" ","")

                if ((tok.dep_ == 'poss') or (tok.dep_.find("obj") == True) or (tok.dep_.find("subj") == True) or (tok.dep_ == "attr"))\
                        and ((tok.text in self.database) or (temp_phrase in self.database)):

                    if prefix != "":
                        self.variable = temp_phrase
                        self.poi = tok.text

                    else:
                        self.variable = self.poi = tok.text

                    if temp_phrase in self.database:
                        self.poi = temp_phrase
                        if tok.dep_ == 'poss':

                            self.string = sent.replace(self.variable + "'s", "%s" % self.database[temp_phrase].label)

                        else:

                            self.string = sent.replace(self.variable, "%s" % self.database[temp_phrase].label)
                    
                    else:
                        if tok.dep_ == 'poss':

                            self.string = sent.replace(self.variable + "'s", "%s" % self.database[tok.text].label)

                        else:

                            self.string = sent.replace(self.variable, "%s" % self.database[tok.text].label)

                if tok.dep_ == "nummod":
                    self.variable += (" " + tok.text) 
                    self.poi = self.variable
                    self.string = sent.replace(self.variable, "[COURSE]")

                if tok.dep_ != "compound" and prefix != "":
                    prefix = ""

                prv_tok_dep = tok.dep_
                prv_tok_text = tok.text
            
        if (self.poi == ""):
            self.poi = self.variable = last_found_var
            self.string = sent.replace(self.variable, "%s" % self.database[last_found_var].label)
            
        print("\n******\nPOI: %s | Variable: %s | String: %s\n******\n" % (self.poi, self.variable, self.string)) 
        returning = [self.poi, self.variable, self.string]
        self.variable = ""
        self.string = ""

        return returning

    def find_answer(self, sent):
        temp_sent = sent
        
        if sent == "The office hours for [PROFESSOR] are [OFFICE HOURS].":

            temp_sent = temp_sent.replace("[PROFESSOR]", self.database[self.poi].name)
            temp_sent = temp_sent.replace("[OFFICE HOURS]", self.database[self.poi].office_hours)

        elif sent == "The prerequisites for [COURSE] are [PREREQUISITES].":
            prereq_str = ""
            for course in self.database[self.poi].prerequisites.split("/"):
                prereq_str += " %s also known as %s," % (course, self.database[course].crosslisted[0])
            temp_sent = temp_sent.replace("[COURSE]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted[0]))
            temp_sent = temp_sent.replace("[PREREQUISITES]", prereq_str)

        elif sent == "[COURSE] is about [INFORMATION].":

            temp_sent = temp_sent.replace("[COURSE]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted[0]))
            temp_sent = temp_sent.replace("[INFORMATION]", self.database[self.poi].information)

        elif sent == "[COURSE] is [UNITS].":
            unit_str = "%s lecture units and %s lab units." % (self.database[self.poi].lecture_units, self.database[self.poi].lab_units)
            temp_sent = temp_sent.replace("[COURSE]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted))
            temp_sent = temp_sent.replace("[UNITS]", unit_str)
        
        elif sent == "[PROFESSOR] is in the [MAJOR] department and teaches [TEACHES].":
            teaches_str = ""
            for course in self.database[self.poi].teaches.split("/"):
                teaches_str += "%s " % course
            temp_sent = temp_sent.replace("[PROFESSOR]", self.database[self.poi].name)
            temp_sent = temp_sent.replace("[MAJOR]", self.database[self.poi].department)
            temp_sent = temp_sent.replace("[TEACHES]", teaches_str)

        elif sent == "[PROFESSOR]'s office is in [BUILDING].":

            temp_sent = temp_sent.replace("[PROFESSOR]", self.database[self.poi].name)
            temp_sent = temp_sent.replace("[BUILDING]", self.database[self.poi].office)

        elif sent == "[PROFESSOR]'s email is [EMAIL].":

            temp_sent = temp_sent.replace("[PROFESSOR]", self.database[self.poi].name)
            temp_sent = temp_sent.replace("[EMAIL]", self.database[self.poi].email)

        elif sent == "[BUILDING] is located at [LOCATION].":

            temp_sent = temp_sent.replace("[BUILDING]", self.database[self.poi].name)
            temp_sent = temp_sent.replace("[LOCATION]", self.database[self.poi].location)

        elif sent == "[CLUB] is about [INFORMATION].":

            temp_sent = temp_sent.replace("[CLUB]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted[0]))
            temp_sent = temp_sent.replace("[INFORMATION]", self.database[self.poi].information)

        elif sent == "[CLUB] meets [MEETING].":

            temp_sent = temp_sent.replace("[CLUB]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted[0]))
            temp_sent = temp_sent.replace("[MEETING]", self.database[self.poi].meeting)

        elif sent == "[CLUB] contact information is [CONTACT INFORMATION].":

            temp_sent = temp_sent.replace("[CLUB]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted[0]))
            temp_sent = temp_sent.replace("[CONTACT INFORMATION]", self.database[self.poi].contact)

        elif sent == "[MAJOR] department office is in [BUILDING].":

            temp_sent = temp_sent.replace("[MAJOR]", self.database[self.poi].name)
            temp_sent = temp_sent.replace("[BUILDING]", self.database[self.poi].department_office)

        elif sent == "[COURSE] is offered [QUARTERS].":

            temp_sent = temp_sent.replace("[COURSE]", "%s (also know as %s)" % (self.database[self.poi].name, self.database[self.poi].crosslisted[0]))
            temp_sent = temp_sent.replace("[QUARTERS]", self.database[self.poi].offered)

        else:
            temp_sent = "Sorry, I do not have an answer for your question"

        self.poi = ""
        return temp_sent

    def select_object_type(self, temp_database, data):
        if temp_database[data]["Type"] == '[COURSE]':
            new_data = Course(name=temp_database[data]["Name"], course_number=temp_database[data]["Course Number"], offered=temp_database[data]["Offered"],\
                major_required=temp_database[data]["Required"], information=temp_database[data]["Information"], major=temp_database[data]["Major"],\
                prerequisites=temp_database[data]["Prerequisites"], corequisites=temp_database[data]["Corequisites"],\
                lecture_units=temp_database[data]["Lecture Units"], lab_units=temp_database[data]["Lab Units"], crosslisted=[word for word in temp_database[data]["Crosslisted"].split("/")])
 
        elif temp_database[data]["Type"] == '[PROFESSOR]':
            new_data = Professor(name=temp_database[data]["Name"], department=temp_database[data]["Department"], teaches=temp_database[data]["Teaches"],
                        office=temp_database[data]["Office"], office_hours=temp_database[data]["Office Hours"], email=temp_database[data]["Email"])

        elif temp_database[data]["Type"] == '[CLUB]':
            new_data = Club(name=temp_database[data]["Name"], information=temp_database[data]["Information"], meeting=temp_database[data]["Meeting"],
                leader=temp_database[data]["Leader"], contact=temp_database[data]["Contact Information"], crosslisted=[word for word in temp_database[data]["Crosslisted"].split("/")])

        elif temp_database[data]["Type"] == '[BUILDING]':
            new_data = Building(name=temp_database[data]["Name"], major=temp_database[data]["Major"], number=temp_database[data]["Number"], 
                    location=temp_database[data]["Location"], crosslisted=[word for word in temp_database[data]["Crosslisted"].split("/")])
        
        elif temp_database[data]["Type"] == '[MAJOR]':
            new_data = Major(name=temp_database[data]["Name"], units=temp_database[data]["Units"], minor=temp_database[data]["Minor"], 
                    grad_program=temp_database[data]["Graduate Program"], department_office=temp_database[data]["Office"], crosslisted=[word for word in temp_database[data]["Crosslisted"].split("/")])
        
        self.database[data] = new_data
   
    def populate_database(self):
        
        with open('db.json', 'r') as data_json:
            temp_database = json.load(data_json)

        for data in temp_database:
            
            if temp_database[data]["Name"] not in self.database:
                self.select_object_type(temp_database, data)

            else:
                self.database[data] = self.database[temp_database[data]["Name"]]
          
            self.add_cross_name(temp_database, self.database[data])


'''
Mock Database Population
'''
class Professor:

    def __init__(self, name=None, department=None, teaches=None,
            office=None, office_hours=None, email=None):
        
        self.label = "[PROFESSOR]"
        self.name = name
        self.department = department
        self.teaches = teaches
        self.office = office
        self.office_hours = office_hours
        self.email = email

class Course:

    def __init__(self, name=None, course_number=None, offered=None,
            major_required=None, information=None, major=None,
            prerequisites=None, corequisites=None, lecture_units=None,
            lab_units=None, crosslisted=None):
        
        self.label = "[COURSE]"
        self.name = name
        self.course_number = course_number
        self.offered = offered
        self.major_required = major_required
        self.information = information
        self.major = major
        self.prerequisites = prerequisites
        self.corequisites = corequisites
        self.lecture_units = lecture_units
        self.lab_units = lab_units
        self.crosslisted = crosslisted

class Major:

    def __init__(self, name=None, units=None, minor=False,
            grad_program=False, department_office=None, crosslisted=None):

        self.label = "[MAJOR]"
        self.name = name
        self.units = units
        self.minor = minor
        self.grad_program = grad_program
        self.department_office = department_office
        self.crosslisted = crosslisted

class Club:

    def __init__(self, name=None, information=None, meeting=None,
            leader=None, contact=None, crosslisted=None):

        self.label = "[CLUB]"
        self.name = name
        self.information = information
        self.meeting = meeting
        self.leader = leader
        self.contact = contact
        self.crosslisted = crosslisted

class Building:

    def __init__(self, name=None, major=None, location=None, number=None, crosslisted=None):
        
        self.label = "[BUILDING]"
        self.name = name
        self.major = major
        self.location = location
        self.number = number
        self.crosslisted = crosslisted

    def __repr__(self):
        return "Building: %s | Major: %s | Location: %s | Number: %s | Crosslisted: %s" % (self.name, self.major, self.location, self.number, self.crosslisted)

if __name__ == '__main__':
    while True:
        var_ext = Variable_Extraction()
        var_ext.populate_database()
        question = input("Enter a question: ")
        answer = input("Enter an answer: ") 
        r = var_ext.extract_variables(question)
        print(var_ext.find_answer(answer))
