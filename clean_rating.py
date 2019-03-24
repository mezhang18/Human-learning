import csv

#Creates new CSV where animes that the user didn't watch (rating==-1) aren't included
input = open('rating.csv', 'r')
output = open('clean_rating.csv', 'w')
writer = csv.writer(output)
count = 0
for row in csv.reader(input):
    if row[2]!="-1":
        writer.writerow(row)
    count += 1
    if count == 2000000:
       break
input.close()
output.close()