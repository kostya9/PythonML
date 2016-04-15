import pandas

contacts = pandas.read_csv("OutlookContacts.csv")
print(contacts.describe())

contacts_new = pandas.DataFrame()

