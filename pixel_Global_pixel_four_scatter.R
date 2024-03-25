#https://stackoverflow.com/questions/37494969/ggplot2-add-regression-equations-and-r2-and-adjust-their-positions-on-plot
#https://stackoverflow.com/questions/7549694/add-regression-line-equation-and-r2-on-graph

library(dplyr) 
library(ggplot2)
library(devtools)
library(ggpmisc)
library(tidyverse)

rm(list=ls())
setwd('J:/SCI6/results/code/TaylorDiagram')
windowsFont(family = "Times New Roman") #define text type
windowsFonts(Times = windowsFont(family = "Times New Roman"))


data <- read_csv("maize.csv")


data$country_ID <- factor(data$country_ID, levels = c('I','II','III','IV','I','II','III','IV','I','II','III','IV','I','II','III','IV')) ##maize

data$model <- factor(data$model, levels = c("WE","SPAM",'LIZUMI'))
data$year <- factor(data$year, levels = c("2000","2005",'2010'))


data=data[data$model ==c('WE'),]


formula <- y ~ x
ggplot(data, aes(x= observe, y=predict , color = year)) +
  geom_point(alpha = 0.3,size=0.5) +
  facet_grid(country_ID~year, scales = "free") +
  geom_abline(intercept=0,slope=1,colour="grey60",linetype="dashed",size=1)+## 1:1 line
  geom_smooth(method = "lm", formula = formula, se = F,size=1) +  
  labs(x = "Reported yield (t/ha)",y = "Predicted yield (t/ha)")+
  theme( legend.position="bottom",legend.direction = "horizontal",
                                                                  legend.background = element_rect(fill="transparent"))+
  stat_poly_eq(aes(label =  ..rr.label..), 
               label.x.npc = "right", label.y.npc = 0.15,
               formula = formula, parse = TRUE, size = 3)



ggsave( file="J:/SCI6/manuscripts/Figure/pixel/country_pixel_maize_scatter.tiff", width=18, height=16,units='cm',dpi =300)








#####################################Global######################################################
rm(list=ls())
setwd('J:/SCI6/results/code/TaylorDiagram')
windowsFont(family = "Times New Roman") #define text type
windowsFonts(Times = windowsFont(family = "Times New Roman"))


data <- read_csv("FourGlobal.csv")


#data$country_ID <- factor(data$country_ID, levels = c('I','II','III','IV','I','II','III','IV','I','II','III','IV','I','II','III','IV')) ##maize

data$model <- factor(data$model, levels = c("WE","SPAM",'LIZUMI'))
data$year <- factor(data$year, levels = c("2000","2005",'2010'))
data$Type <- factor(data$Type, levels = c("Maize","Rice",'Wheat',"Soybean"),labels = c("Maize","Rice",'Wheat',"Soybean"))


data=data[data$model ==c('SPAM'),]


formula <- y ~ x
ggplot(data, aes(x= observe, y=predict , color = year)) +
  geom_point(alpha = 0.3,size=0.5) +
  facet_grid(year~Type, scales = "free") +
  geom_abline(intercept=0,slope=1,colour="grey60",linetype="dashed",size=1)+## 1:1 line
  geom_smooth(method = "lm", formula = formula, se = F,size=1) +  
  labs(x = "Reported yield (t/ha)",y = "Predicted yield (t/ha)")+
  theme( legend.position="bottom",legend.direction = "horizontal",
         legend.background = element_rect(fill="transparent"))+
  stat_poly_eq(aes(label =  ..rr.label..), 
               label.x.npc = "right", label.y.npc = 0.15,
               formula = formula, parse = TRUE, size = 3)



ggsave( file="J:/SCI6/manuscripts/Figure/pixel/Global_pixel_four_scatter_SPAM.tiff", width=18, height=16,units='cm',dpi =300)







