import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.javacpp.opencv_highgui.*;

/*
Name: Md Kamrul Hasan
Email: hasan.alive@gmail.com
*/


public class VideoProcessing {


    //private static Logger log = Logger.getAnonymousLogger();

    private static final String AUTONOMOUS_DRIVING = "Object Detection(TU Berlin)";
    private String windowName;
    private volatile boolean stop = false;
    static SavedModelBundle model ;

   // private final OpenCVFrameConverter.ToMat conv = new OpenCVFrameConverter.ToMat();
    private final OpenCVFrameConverter.ToMat convert = new OpenCVFrameConverter.ToMat();
    public final static AtomicInteger atomicInteger = new AtomicInteger();

    DetectObjects detobj = new DetectObjects();

    public void startRealTimeVideoDetection(Session models, String[] labels,String videoFileName) throws java.lang.Exception {
        //log.info("Start detecting video " + videoFileName);

        int id = atomicInteger.incrementAndGet();
        windowName = AUTONOMOUS_DRIVING + id;
      //  log.info(windowName);




        //model = SavedModelBundle.load(models, "serve");

        //Session sees = model.session();
        Session sees = models;
         //sees.runner();


        startYoloThread(sees,labels);
        runVideoMainThread(videoFileName, convert,labels,sees);
        sees.close();
    }

    private void runVideoMainThread(String videoFileName, OpenCVFrameConverter.ToMat convert,String [] labels,Session sees) throws IOException, FrameGrabber.Exception {
      //  File videoFile = new File(videoFileName);
        FFmpegFrameGrabber grabber = initFrameGrabber(videoFileName);
        while (!stop) {
            Frame frame = grabber.grab();
            if (frame == null) {
               // log.info("Stopping");
                stop();
                break;
            }
            if (frame.image == null) {
                continue;
            }
            detobj.push(frame);
            opencv_core.Mat mat = convert.convert(frame);
           // Mat mat = toMat.convert(frame);



            detobj.drawBoundingBox(frame, mat,labels,sees);
            //imshow(windowName,mat);

            imshow(windowName,mat);
            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                stop();
                break;
            }
        }
    }

    private FFmpegFrameGrabber initFrameGrabber(String videoFileName) throws FrameGrabber.Exception {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFileName);
        grabber.start();
        return grabber;
    }

    private void startYoloThread(Session sess , final String[] labels) {

        Thread thread = new Thread(() -> {
            while (!stop) {
                try {
                    detobj.detectObjects(labels,sess);
                } catch (Exception e) {
                    //ignoring a thread failure
                    //it may fail because the frame may be long gone when thread get chance to execute
                }
            }
            detobj = null;
           // log.info("MobileNet Thread Exit");
        });
        thread.start();
    }

    public void stop() {
        if (!stop) {
            stop = true;
            destroyAllWindows();
        }
    }



}
