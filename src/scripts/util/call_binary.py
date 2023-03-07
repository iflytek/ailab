import subprocess
from subprocess import Popen, PIPE
import time
import queue
import threading


class QueueCmdInteractor:
    def __init__(self, cmd, outq=None):
        if not outq:
            outq = queue.Queue()
        self.p = Popen(cmd, shell=False, stdin=PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.outq = outq
        t = threading.Thread(target=self.output_reader, args=(self.p, outq))
        t.start()

    def sendline(self, msg):
        self.p.stdin.write(("%s\n" % (msg)).encode());
        self.p.stdin.flush()

    def readline(self, timeout=10):
        stop = False
        while not stop:
            line = self.outq.get(timeout)

    def output_reader(self, proc, outq):
        for line in iter(proc.stdout.readline, b''):
            outq.put(line.decode('utf-8'))

    def __del__(self):
        self.p.terminate()


class FileCmdInteractor:
    def __init__(self, cmd):
        import tempfile
        self.cmd = cmd
        self.fw = tempfile.NamedTemporaryFile()
        self.fr = open(self.fw.name, "r")
        self.p = Popen(cmd, shell=True, stdin=PIPE, stdout=self.fw, stderr=self.fw)

    def sendline(self, msg):
        self.p.stdin.write(("%s\n" % (msg)).encode());
        self.p.stdin.flush()

    def readline(self, timeout=10):
        r'''  
        Read one line from stdout  

        @param timeouot:  
            Timeout in second  
        '''

        out = self.fr.readline()
        wi = 0
        while out is None or out == '':
            wi = wi + 1
            if wi > 10:
                raise Exception('Timeout in reading line from output!')
            time.sleep(1)
            out = self.fr.readline()
        return out.rstrip()

    def __del__(self):
        self.fw.close()
        self.fr.close()
        self.p.terminate()


if __name__ == '__main__':
    ci = FileCmdInteractor('bash')
    ci.sendline('ls \n')
    print("Read: %s" % (ci.readline()))
