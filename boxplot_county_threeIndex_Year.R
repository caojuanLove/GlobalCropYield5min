rm(list=ls())
setwd("J:/SCI6/results/pixel_validation")
library(ggplot2)
library(readxl)
windowsFont(family = "Times New Roman") #define text type
windowsFonts(Times = windowsFont(family = "Times New Roman"))
DATA<-read.table("allfour_index.csv",sep=',',header=TRUE)

DATA=DATA[DATA$variable !=c('mae'),]

DATA$NAME_0 <- factor(DATA$country_ID, levels = c('I','II','III','IV','V','VI','VII','VIII')) ##maize
DATA$Type <- factor(DATA$Type, levels = c("Maize","Rice",'Wheat',"Soybean"),labels = c("Maize","Rice",'Wheat',"Soybean"))
DATA$variable <- factor(DATA$variable, levels = c("r2","rmse",'nrmse'))
DATA$variable <- factor(DATA$variable,labels = c("R^{2}","RMSE", "NRMSE"))


data_mean <-aggregate(x= DATA$value,by=list(DATA$Type,DATA$variable,DATA$year),mean)

write.csv(data_mean,'D:/manuscripts/table/pixel/data_mean_pixel.csv')


DATA=DATA[DATA$variable ==c('NRMSE'),]

p3<-ggplot(DATA,aes(x =year , y =value ,group = year,fill= Type),)+ ##指定按什么颜色来区分颜色
  #ggplot(aes(x=x_int, y=y, group = interaction(x_int, class), fill = class)) , group = interaction(year, Type),fill= Type
  geom_boxplot( show.legend = TRUE,width = 0.5,outlier.shape = NA)+
  stat_summary(fun=mean, geom="point", shape=20, size=2.5, color="red", 
               position = position_dodge2(width = 0.53, preserve = "single")) +
  labs(x = "",y = "")+
  xlim(1981,2016)+
  scale_x_continuous(breaks = c(1985, 1995, 2005,2015))+
  theme( legend.position="bottom",legend.direction = "horizontal",
                      legend.background = element_rect(fill="transparent"))+
  guides(fill=guide_legend(title=NULL))+
  # 
  theme(axis.text = element_text(size=10, family = "Times",color="black"),line=element_line(color='grey',size=0.5))+
  theme(panel.grid.major = element_line(color='grey',linetype = "dashed"),
                      panel.grid.minor = element_line(color='grey',linetype = "dashed"),
                      strip.text = element_text(size=10, family = "Times",color="black"),
                      panel.background = element_rect(fill='transparent', color='black'))+
  facet_grid(Type~., scales="free")#+
p3

ggsave(p3, file="J:/SCI6/manuscripts/Figure/year_pixel_NRMSE.tiff", width=16, height=18,units='cm',dpi =300)



  # facetted_pos_scales( y = list(Type == "Maize" ~ scale_y_continuous(breaks = c(0.3, 0.5, 0.7,0.9)),
  #                               Type == "Rice" ~ scale_y_continuous(breaks = c( 0.4, 0.6,0.8)),
  #                               Type == "Wheat" ~ scale_y_continuous(breaks = c(0.4, 0.6, 0.8)),
  #                               Type == "Soybean" ~ scale_y_continuous(breaks = c(0.4, 0.6, 0.8, 1.0))
  #                              ))#
  # 

#ggsave(p3, file="J:/SCI6/manuscripts/Figure/country_pixel_time_NRMSE.tiff", width=16, height=16,units='cm',dpi =300)