from aiosmtpd.controller import Controller
import asyncio

class Handler:
    async def handle_DATA(self, server, session, envelope):
        print('\n--- EMAIL RECEIVED ---')
        print('From:', envelope.mail_from)
        print('To:', envelope.rcpt_tos)
        try:
            print('Content:\n')
            print(envelope.content.decode('utf8', errors='replace'))
        except Exception as e:
            print('Could not decode content', e)
        return '250 Message accepted for delivery'

if __name__ == '__main__':
    controller = Controller(Handler(), hostname='127.0.0.1', port=1025)
    controller.start()
    print('Local SMTP debug server running on 127.0.0.1:1025')
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        controller.stop()
