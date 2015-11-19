import urllib2
import re

def url_input():
  url_input = raw_input("Please Enter your URL: ")
  return url_input

def get_html_response(url_input):
  html_response = urllib2.urlopen(url_input)
  html_read = html_response.read()
  return html_read

def find_url_in_response(html_read):
  find_urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', html_read)
  return find_urls

input = url_input()
response = get_html_response(input)
find = find_url_in_response(response)
output = find
print output

