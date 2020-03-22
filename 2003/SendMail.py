def send_mail(subject="제목" ,**kwargs) :
    """
    kwrags : txtpath , imgpath , gifpath [list]
    """
    import os , re
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    import codecs
    import sys

    TEXT = """
    결과물을 확인해주세요!
    Jupyter Notebook을 종료합니다. 
    {txt}
    """
    if "txt" in kwargs:
        TEXT.format(txt=kwargs["txt"])
    sender_email = 'leesungreong@gmail.com' #'‘sendermail@example.com’       # 송신 메일
    receiver_email = 'leesungreong@gmail.com'        # 수신 메일
    login = 'leesungreong@gmail.com'
    password = 'tjdfud65631!1'
    msg = MIMEMultipart('SendMail')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    part2 = MIMEText(TEXT)
    msg.attach(part2)
    if "txtpath" in kwargs :
        filenamelists = kwargs["txtpath"]
        if isinstance(filenamelists, list):  pass
        else : filenamelists = list(filenamelists)
        for filename in filenamelists :
            f = codecs.open(filename, 'rb', 'utf-8')
            attachment = MIMEText(f.read())
            attachment.add_header('Content-Disposition', 'attachment',
                                  filename=filename.split("/")[-1])
            msg.attach(attachment)
    if "gifpath" in kwargs :
        filenamelists = kwargs["gifpath"]
        if isinstance(filenamelists, list):  pass
        else : filenamelists = list(filenamelists)
        for filename in filenamelists :
            f = codecs.open(filename, 'rb')
            attachment = MIMEImage(f.read())
            attachment.add_header('Content-Disposition',
                                  'attachment', 
                                  filename=filename.split("/")[-1])           
            msg.attach(attachment)
    if "imgpath" in kwargs :
        filenamelists = kwargs["imgpath"]
        if isinstance(filenamelists, list):  pass
        else : filenamelists = list(filenamelists)
        for filename in filenamelists :
            img_data = open(filename, 'rb').read()
            image = MIMEImage(img_data, name=filename.split("/")[-1])
            msg.attach(image)    
    with smtplib.SMTP_SSL("smtp.gmail.com") as server:
        server.login(login, password)
        server.sendmail(sender_email,
                        receiver_email,
                        msg.as_string())
    #sys.exit(0)