import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.javacpp.opencv_highgui.*;


import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;


public class VideoProcessing {


    //private static Logger log = Logger.getAnonymousLogger();

    private static final String AUTONOMOUS_DRIVING = "Autonomous Driving(TU Berlin)";
    private String windowName;
    private volatile boolean stop = false;

   // private final OpenCVFrameConverter.ToMat conv = new OpenCVFrameConverter.ToMat();
    private final OpenCVFrameConverter.ToMat convert = new OpenCVFrameConverter.ToMat();
    public final static AtomicInteger atomicInteger = new AtomicInteger();

    DetectObjects detobj = new DetectObjects();

    public void startRealTimeVideoDetection(String model, String[] labels,String videoFileName) throws java.lang.Exception {
        //log.info("Start detecting video " + videoFileName);

        int id = atomicInteger.incrementAndGet();
        windowName = AUTONOMOUS_DRIVING + id;
      //  log.info(windowName);



        detobj.modelbuild(model);
        startYoloThread(model,labels);
        runVideoMainThread(videoFileName, convert);
    }

    private void runVideoMainThread(String videoFileName, OpenCVFrameConverter.ToMat convert) throws IOException, FrameGrabber.Exception {
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



            detobj.drawBoundingBox(frame, mat);
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

    private void startYoloThread(final String model, final String[] labels) {

        Thread thread = new Thread(() -> {
            while (!stop) {
                try {
                    detobj.detectObjects(labels);
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
