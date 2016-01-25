import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
 
#def email():
fromaddr = "vagrantscript@gmail.com"
toaddr = "ib2g14@soton.ac.uk"
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Finished"
 
body = "Your script has finished"
msg.attach(MIMEText(body, 'plain'))
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "Vagrant123")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()
#return
#email()