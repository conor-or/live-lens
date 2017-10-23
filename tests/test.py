from Tkinter import *

class MyDialog:

    def __init__(self, parent):

        top = self.top = Toplevel(parent)

        Label(top, text='value').pack()

        self.e = Entry(top)
        self.e.pack(padx=5)

        b = Button(top, text='ok', command=self.ok)
        b.pack(pady=5)

    def ok(self):

        print 'value is', self.e.get()
        self.top.destroy()

root = Tk()

Button(root, text='hello').pack()

root.update()

d = MyDialog(root)
root.wait_window(d.top)
