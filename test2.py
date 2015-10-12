# coding=utf-8
from pyinference.inference import Factor, Variable, Net
import numpy as np


The_results_of_learning_activities = Variable (name= 'The results of learning activities' , terms=['low', 'middle', 'high'])
Results_of_Olympiads = Variable (name= 'Results of Olympiads' , terms=['low', 'middle', 'high'])
Health = Variable (name= 'Health' , terms=['bad', 'medium', 'high'])
Education_Level = Variable (name= 'Education Level' , terms=['low', 'middle', 'high'])

Education_Level_F = Factor (cons=[Education_Level], cond=[The_results_of_learning_activities, Results_of_Olympiads, Health])
Education_Level_F.cpd = np.array ([1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                                   1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
                                   0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,]).reshape((3, 3, 3, 3))

Additional_education = Variable (name= 'Additional education' , terms=['no', 'yes'])
The_organization_of_the_educational_process = Variable(name= 'The organization of the educational process' , terms=['With minor violations of', 'According to the norms'])
Qualifications_of_teachers = Variable(name= 'Qualifications of teachers' , terms=['Most teachers without a category' , 'Most teachers 1 a category' , 'Most teachers of the highest category'])
Scientific_methodical_activity_of_teachers = Variable(name= 'Scientific-methodical activity of teachers' , terms=['low', 'middle', 'high'])
Educational_environment = Variable (name= 'Educational environment', terms=['low', 'middle', 'high'])

Educational_environment_F = Factor (cons= [Educational_environment], cond=[Additional_education, The_organization_of_the_educational_process, Qualifications_of_teachers, Scientific_methodical_activity_of_teachers, Educational_environment])
Educational_environment_F.cpd = np.array ([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                                           1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                                           0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                           0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,]).reshape((2, 2, 3, 3, 3))

Education_Level = Variable(name='Education level', terms=['low', 'middle', 'high'])
Educational_environment = Variable(name='Educational environment', terms=['low', 'middle', 'high'])
School_rating = Variable(name='School rating', terms=['low', 'middle', 'high'])

School_rating_F = Factor(cons=[School_rating], cond=[Education_Level, Educational_environment])
School_rating_F.cpd = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0,
                                0, 1, 0, 0, 1, 0, 0, 0, 1,
                                0, 0, 1, 0, 0, 1, 0, 0, 1]).reshape((3, 3, 3))


Geographical_position = Variable (name= 'Geographical position', terms=['district', 'region'])
The_level = Variable (name= 'The level' , terms=['high school', 'correctional school', 'specialized school'])
Theoretical_demand = Variable (name= 'Theoretical demand' , terms=['low', 'middle', 'high'])
Theoretical_demand_F = Factor (cons= [Theoretical_demand], cond=[Geographical_position, The_level])
Theoretical_demand_F.cpd = np.array ([0, 1, 0, 1, 0, 0, 0, 1, 0,
                                      0, 0, 1, 1, 0, 0, 0, 0, 1]).reshape((2, 3, 3))

Theoretical_demand = Variable (name= 'Geographical position', terms=['low', 'middle', 'high'])
School_rating = Variable(name='School rating', terms=['low', 'middle', 'high'])
Number_of_trainees_in_new_educational_year = Variable(name='Number of trainees in new educational year', terms=['Less than 350 people', 'Less than 650 people', 'More than 650 people'])
Number_of_trainees_in_new_educational_year_F = Factor (cons = [Number_of_trainees_in_new_educational_year], cond=[Theoretical_demand, School_rating])
Number_of_trainees_in_new_educational_year_F.cpd = np.array ([1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                              0, 1, 0, 0, 1, 0, 0, 0, 1,
                                                              0, 1, 0, 0, 0, 1, 0, 0, 1,]).reshape((3, 3, 3))

BN = Net(name='Number_of_trainees', nodes=[Education_Level_F, Educational_environment_F, School_rating_F, Theoretical_demand_F, Number_of_trainees_in_new_educational_year_F])












