from tkinter.messagebox import showinfo
import tkinter
import PIL.Image, PIL.ImageTk
import pickle
import os
import imageio
import torchvision

class App:
    def __init__(self, window, window_title, video_source, database, mrcnn_api,gaitnet_api):
        self.database = database
        self.mrcnn_api, self.gaitnet_api = mrcnn_api,gaitnet_api

        self.is_record = False

        self.frames = []
        self.segs = []

        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = self.video_source

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.btn_capture=tkinter.Button(window, text="Capture", width=50, command=self.record)
        self.btn_save = tkinter.Button(window, text="Save", width=50, command=self.save)
        self.l1 = tkinter.Label(self.window, text='Name')
        self.e1 = tkinter.Entry(self.window)

        self.canvas.pack()
        self.l1.pack()
        self.e1.pack()
        self.btn_capture.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_save.pack(anchor=tkinter.CENTER, expand=True)
        self.delay = 1
        self.update()
        self.index = 0
        self.window.mainloop()

    def save(self):
        name = self.e1.get()
        if name == '':
            showinfo("Error", "Please enter a name")
            return

        try:
            self.segs  = self.mrcnn_api.get_seg_batch(self.frames,10)
            feature = self.gaitnet_api.main(self.segs) # from video to feature
        except:
            print("Video processing failed. Please record again.")
            self.frames.clear()
            self.is_record = True
            return

        self.database[name] = feature # save feature in database
        if not os.path.exists('database/{}'.format(name)):
            os.makedirs('database/{}'.format(name))
        else:
            filelist = [f for f in os.listdir('database/{}'.format(name))]
            for f in filelist:
                os.remove(os.path.join('database/{}'.format(name), f))

        for i, f in enumerate(self.segs):
            torchvision.utils.save_image(f,os.path.join('database/{}/{:03d}.png'.format(name, i)))
            # imageio.imwrite()

        print(name,"is saved")
        with open('database/data.pickle', 'wb') as handle:
            pickle.dump(self.database, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.frames.clear()
        self.is_record = True

    def record(self):
        if self.btn_capture['text'] == "Capture":
            self.btn_capture['text'] = 'Stop'
            # events
            self.frames.clear()
            self.is_record = True
        else:
            self.btn_capture['text'] = 'Capture'
            self.is_record = False

    def update(self):
        if self.is_record is False and len(self.frames) != 0:  # replay
            if self.index>=len(self.frames):
                self.index = 0
            frame = self.frames[self.index]
            self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=tkinter.NW)
            self.index+=1
        else:  # live
            exist, frame = self.vid.get_frame()
            if exist:  # if get the frame
                self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.frame, anchor=tkinter.NW)
                if self.is_record: # if error raised by above line
                    self.frames.append(frame)
        self.window.after(self.delay, self.update)

def main(database,video_source, mrcnn_api,gaitnet_api, args__):
    App(tkinter.Toplevel(), "GaitNet Enrollment",video_source, database,mrcnn_api,gaitnet_api)