import webbrowser

def type_in_your_friends_name():
	friend_name = raw_input('Please type in the user name you are looking for: ')
        return friend_name

def find_friend_in_hash(friend_name):
	users = {'Max': 3,'Tim': 4, 'Abraoo': 5}
        friends = users.get(friend_name)
	return friends
		
def output_the_friend_result(friends):
        url = "file:/python_exercises/output.html"
        if friends <= 0:
		print "forever alone"
	else:
	        webbrowser.open(url)

input = type_in_your_friends_name()
find = find_friend_in_hash(input)
output = output_the_friend_result(find)
