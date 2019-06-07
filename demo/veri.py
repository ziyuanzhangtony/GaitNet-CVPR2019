from tkinter.messagebox import showinfo
import tkinter
import PIL.Image, PIL.ImageTk
from utils.helper import calculate_cosine_similarity

threshold = 0.85
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
        # self.canvas2 = tkinter.Canvas(window, width=128, height=256)
        self.btn_capture=tkinter.Button(window, text="Capture", width=50, command=self.snapshot)
        self.btn_save = tkinter.Button(window, text="Verify", width=50, command=self.verifi)


        self.l1 = tkinter.Label(self.window, text='Name')
        self.e1 = tkinter.Entry(self.window)
        # e1.grid(row=0, column=1)

        self.canvas.pack()
        # self.canvas2.pack()
        self.l1.pack()
        self.e1.pack()
        self.btn_capture.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_save.pack(anchor=tkinter.CENTER, expand=True)


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.index = 0

        self.window.mainloop()

    def verifi(self):
        name = self.e1.get()
        if name == '':
            showinfo("Error", "Please enter a name to verify")
            return
        elif name not in self.database:
            showinfo("Error", "Name is NOT registered in database")
            return
        if self.database == {}:
            showinfo("Error", "Database is empty!")
            return
        try:
            self.segs = self.mrcnn_api.get_seg_batch(self.frames, 20)
            feature = self.gaitnet_api.main(self.segs)  # from video to feature
        except:
            print("Video processing failed. Please record again.")
            self.frames.clear()
            self.is_record = True
            return
        feature__ = self.database[name] #  get feature from database via name
        score = calculate_cosine_similarity(feature,feature__)
        if score >= threshold:
            showinfo("INFO", "YES, this is {}. Confidence:{}".format(name,round(score*100,2)))
        else:
            showinfo("INFO", "NO, this is NOT {}. Confidence:{}".format(name, round(score*100,2)))

    def snapshot(self):
        # Get a frame from the video source
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
            if self.index >= len(self.frames):
                self.index = 0
            frame = self.frames[self.index]
            self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=tkinter.NW)
            self.index += 1
        else:  # live
            exist, frame = self.vid.get_frame()
            if exist:  # if get the frame
                self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.frame, anchor=tkinter.NW)
                if self.is_record: # if error raised by above line
                    self.frames.append(frame)

        self.window.after(self.delay, self.update)


def main(database,video_source, mrcnn_api,gaitnet_api, a):
    global threshold
    threshold = a.threshold
    App(tkinter.Toplevel(), "GaitNet Verification",video_source, database,mrcnn_api,gaitnet_api)