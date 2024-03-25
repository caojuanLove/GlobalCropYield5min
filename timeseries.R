

###########################rice##############################
rm(list=ls())
setwd("J:/SCI6/results/Table")
library(readxl)
library(ggplot2)
windowsFont(family = "Times New Roman") #define text type
windowsFonts(Times = windowsFont(family = "Times New Roman"))
library(ggh4x)
data <- read_excel(path =  "fourcrop.xlsx",sheet='cropall')
data=data[data$Type ==c('Record'),]
data=data[data$crop ==c('Rice'),]
data$country_ID <- factor(data$country_ID, levels = c('I','II','III','IV')) ##maize
data$Lable <- factor(data$country_ID, levels = c('FAO','Actual','Predict')) ##maize
#data$crop <- factor(data$crop, levels = c('Maize','Rice','Wheat','Soybean')) ##maize
p3<-ggplot(data)+ ##指定按照什么颜色来区分颜色+
  geom_line(aes(x = year, y = value, colour = Label,linetype =Label )) +
  geom_point(aes(x = year, y = value, colour = Label))+
  facet_grid(country_ID~., scales="free") +# Type
  labs(x = "",y = 'Yield (t/ha)')+
  guides(fill=guide_legend(title=NULL))+
  theme(legend.justification=c(1,0), legend.position=c(0.97,0.74),
        legend.direction = "horizontal",
        legend.background = element_rect(fill="transparent"),
        legend.title = element_blank())+
  theme(legend.key = element_blank())+
  # ylim(0,1)
  theme(axis.text = element_text(size=12, family = "Times",color="black"),line=element_line(color='grey',size=0.5))+
  theme(panel.grid.major = element_line(color='grey',linetype = "dashed"),
        panel.grid.minor = element_line(color='grey',linetype = "dashed"),
        strip.text = element_text(size=14, family = "Times",color="black"),
        panel.background = element_rect(fill='transparent', color='black'))+
  scale_x_continuous(breaks = c(1985, 1995, 2005,2015))+
  theme(strip.text = element_blank())+
  facetted_pos_scales( y = list(country_ID == "I" ~ scale_y_continuous(breaks = c(5, 6, 7)),
                                country_ID == "II" ~ scale_y_continuous(breaks = c(2, 3, 4)),
                                country_ID == "III" ~ scale_y_continuous(breaks = c(3, 4, 5)),
                                country_ID == "IV" ~ scale_y_continuous(breaks = c(2, 3, 4))))
p3

ggsave(p3, file="J:/SCI6/manuscripts/Figure/Rice1.tiff", width=18.5, height=9,units='cm',dpi =300)
















##########################wheat##########################
rm(list=ls())
setwd("J:/SCI6/results/Table")
library(readxl)
library(ggplot2)
windowsFont(family = "Times New Roman") #define text type
windowsFonts(Times = windowsFont(family = "Times New Roman"))
library(ggh4x)
data <- read_excel(path =  "fourcrop.xlsx",sheet='cropall')
data=data[data$Type ==c('Record'),]
data=data[data$crop ==c('Wheat'),]
data$country_ID <- factor(data$country_ID, levels = c('I','II','III','IV')) ##maize
data$Lable <- factor(data$country_ID, levels = c('FAO','Actual','Predict')) ##maize
#data$crop <- factor(data$crop, levels = c('Maize','Rice','Wheat','Soybean')) ##maize
p3<-ggplot(data)+ ##指定按照什么颜色来区分颜色+
  geom_line(aes(x = year, y = value, colour = Label,linetype =Label )) +
  geom_point(aes(x = year, y = value, colour = Label))+
  facet_grid(country_ID~., scales="free") +# Type
  labs(x = "",y = 'Yield (t/ha)')+
  guides(fill=guide_legend(title=NULL))+
  theme(legend.justification=c(1,0), legend.position=c(0.97,0.74),
        legend.direction = "horizontal",
        legend.background = element_rect(fill="transparent"),
        legend.title = element_blank())+
  theme(legend.key = element_blank())+
  # ylim(0,1)
  theme(axis.text = element_text(size=12, family = "Times",color="black"),line=element_line(color='grey',size=0.5))+
  theme(panel.grid.major = element_line(color='grey',linetype = "dashed"),
        panel.grid.minor = element_line(color='grey',linetype = "dashed"),
        strip.text = element_text(size=14, family = "Times",color="black"),
        panel.background = element_rect(fill='transparent', color='black'))+
  scale_x_continuous(breaks = c(1985, 1995, 2005,2015))+
  theme(strip.text = element_blank())+
  facetted_pos_scales( y = list(country_ID == "I" ~ scale_y_continuous(breaks = c(6, 8, 10)),
                                country_ID == "II" ~ scale_y_continuous(breaks = c(4, 5, 6)),
                                country_ID == "III" ~ scale_y_continuous(breaks = c(2, 3, 4)),
                                country_ID == "IV" ~ scale_y_continuous(breaks = c(3, 5, 7))))
p3
ggsave(p3, file="J:/SCI6/manuscripts/Figure/wheat1.tiff", width=18.5, height=9,units='cm',dpi =300)




