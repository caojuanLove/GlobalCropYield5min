rm(list=ls())
setwd("J:/SCI6/results/pixel_validation")
library(ggplot2)
windowsFont(family = "Times New Roman") #define text type
windowsFonts(Times = windowsFont(family = "Times New Roman"))
DATA<-read.table("allfour_index.csv",sep=',',header=TRUE)
DATA=DATA[DATA$variable !=c('mae'),]
DATA$NAME_0 <- factor(DATA$country_ID, levels = c('I','II','III','IV','V','VI','VII','VIII')) ##maize
DATA$Type <- factor(DATA$Type, levels = c("Maize","Rice",'Wheat',"Soybean"))
DATA$variable <- factor(DATA$variable, levels = c("r2","rmse",'nrmse'))
DATA$variable <- factor(DATA$variable,labels = c("R^{2}","RMSE", "NRMSE"))

data_mean <-aggregate(x= DATA$value,by=list(DATA$Type,DATA$country_ID,DATA$variable),mean)

write.csv(data_mean,'J:/SCI6/results/pixel_validation/data_mean.csv')

p3<-ggplot(DATA,aes(x = country_ID, y = value, fill= DATA$Type))+ ##指定按照什么颜色来区分颜色
  geom_boxplot(show.legend = TRUE,width = 0.5,outlier.shape = NA)+
  stat_summary(fun=mean, geom="point", shape=20, size=2.5, color="red", 
               position = position_dodge2(width = 0.53, preserve = "single")) +
  labs(x = "",y = "")+
  theme(legend.justification=c(1,0), legend.position=c(1,0.68),
                      legend.background = element_rect(fill="transparent"))+
  guides(fill=guide_legend(title=NULL))+
  # ylim(0,1)+
  theme(axis.text = element_text(size=12, family = "Times",color="black"),line=element_line(color='grey',size=0.5))+
  theme(panel.grid.major = element_line(color='grey',linetype = "dashed"),
                      panel.grid.minor = element_line(color='grey',linetype = "dashed"),
                      strip.text = element_text(size=14, family = "Times",color="black"),
                      panel.background = element_rect(fill='transparent', color='black'))+
  facet_grid(variable~., scales="free",labeller = label_parsed) # Type#,labeller =my_labels
p3
ggsave(p3, file="J:/SCI6/manuscripts/Figure/country_pixel.tiff", width=16, height=18,units='cm',dpi =300)