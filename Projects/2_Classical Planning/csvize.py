dump = open('experiments.txt')
out = open('experiments2.csv', 'w+')

out.write("problem,algorithm,|actions|,|expansions|,|goal tests|,|new nodes|,plan length,time\n")

i = 100
for line in dump:
	if line.startswith('Solving'):
		problem_number = line[26]
		algorithm = line[34:-3]
		i = 0
	elif i == 3:
		actions, expansions, goal_tests, new_nodes = line.split()
	elif i == 5:
		words = line.split()
		plan_length = words[2]
		time = words[7]
		out.write(problem_number+','+algorithm+','+actions+','+expansions+','+goal_tests+','+
				new_nodes+','+plan_length+','+time+'\n')
	i += 1

