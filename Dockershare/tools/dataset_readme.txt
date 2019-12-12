DeviceNum=3
DeviceName="CavyBand(Cavy2-A8B1)","CavyBand(Cavy2-A5B2)","CavyBand(Cavy2-1A3A)"
ActionListFile=gesture.txt
ColumnDescription=Timestamp,Quaternions(1),Euler Angle(1),Linear Accelerometer(1),Velocity(1),Quaternions(2),Euler Angle(2),Linear Accelerometer(2),Velocity(2),Quaternions(3),Euler Angle(3),Linear Accelerometer(3),Velocity(3),Label,,Gravity Accelerometer(1),Body Accelerometer(1),Gravity Accelerometer(2),Body Accelerometer(2),Gravity Accelerometer(3),Body Accelerometer(3)
ColumnDim=1,4,3,3,3,4,3,3,3,4,3,3,3,1,3,3,3,3,3,3
FrameRate=59



註：
1.資料內容是由3個感測裝置測得數值之集合，部分ColumnDescription後的(n)表第n個感測器，如：Quaternions(1)表第1個感測器量測的Quaternions(四元數)數值
2.此資料集已完成單次動作資料劃分、time stamp alignment、濾出Gravity Accelerometer和Body Accelerometer的動作
3.ColumnDim表各項目維度，與ColumnDescription項目一一對應
4.dataset中總共有47類動作，部分動作並未收集
5.每個csv檔即是單次動作的開始至結束