#https://waterprogramming.wordpress.com/2020/12/22/taylor-diagram/

library(openair)
library(tidyverse)

getwd()
setwd("J:/SCI6/results/code/TaylorDiagram")

SPIscale =1

data <- paste("./maize7To3",".csv",sep = "")
mydata <-read.csv(data, header = T)

# mydata$year <- factor(mydata$year,order =TRUE, levels = c("2000","2005","2010"),
#                       labels = c("2000","2005","2010"))
mydata$date <- strptime(paste(mydata$year,"1", "1",sep = "-"), format = "%Y-%m-%e")
mydata$date <- as.POSIXct(mydata$date)

datawe = mydata[which(mydata$model == 'WE'),]
TaylorDiagram(datawe, obs = "observe", mod = "predict", group = "NAME_0", type = "year")







#mydata <- mydata[,-c(1,2)]

#对class进行排序以绘图
#自定义排序https://www.codenong.com/46129322/
mydata$month <- factor(mydata$month,order =TRUE, levels = c("1","2","3","4","5","6","7","8","9","10","11","12"),
                       labels = c("Jan", 	"Feb", 	"Mar", 	"Apr", 	"May", 	"Jun", 	"Jul", 	"Aug", 	"Sep", 	"Oct", 	"Nov", 	"Dec")
)
mydata$model <- factor(mydata$model,order =TRUE, levels = c("CHIRPS","ERA","FLDAS","PERSI","TERRA","GPR"))

# 分月进行统计

jpeg(file = paste("./figures/SPI_month_", SPIscale, ".jpg",sep = ""),width=21,height=25,res = 300, units = 'cm')#或者png修改jpeg

TaylorDiagram(mydata, obs = "observe", mod = "predict", group = "model",type = "month",
              #cols = c("#d7191c", "#0099CC","#79bd9a")
              key.pos = "bottom",key.columns = 3,key.title = 'Model',
              #col=c("#fdae61", "#8073ac","#1b7837","#4393c3","#8dd3c7","#d73027"),
              cols = "Paired",
              #normalise = TRUE,
              pch = c(15, 15, 15,15,15,15),
              annotate = "RMSE"
              #cex = c(1.7,1.6,1.4,1.5,1.1,1.5),
              #layout = c(4, 3)
)

#TaylorDiagram(mydata, obs = "observe", mod = "predict", group = "model",type = "season")
dev.off()

# 统计所有年，所有月的

#导出图片格式
jpeg(file = paste("./figures/SPI_all_", SPIscale, ".jpg",sep = ""),width=2000,height=1500,res = 300, units = 'px')#或者png修改jpeg

TaylorDiagram(mydata, obs = "observe", mod = "predict", group = "model",
              key.columns = 3, key.pos = "bottom",key.title = 'Model ',
              cols = "Paired",
              pch = c(15, 15, 15,15,15,15),
              #normalise = TRUE,
              annotate = "RMSE"
              #cex = c(1.7,1.7,1.5)
)
dev.off()
# 
# #导出PDF格式
# pdf("../figures/Tmean_GPR_Terra_allMonth.pdf") 
# 
# TaylorDiagram(mydata, obs = "observe", mod = "predict", group = "model",
#               key.columns = 3, key.pos = "bottom",key.title = 'Tmean ',
#               pch = c(20, 20, 18),cex = c(1.7,1.7,1.5))
# 
# dev.off()




