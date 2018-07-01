library(dplyr)
library(stringr)
library(ggplot2)


setwd("C:/Users/martin.xie/PycharmProjects/ParetoQLearning/data/log")

#mazex <- c('maze6_path_weight100_weight30.csv','data.20180630-181815.csv') 
mazex <- c('maze6_path_weight100_weight30.csv','qlMaze.data.20180702-113953.csv')#qlMaze.data.20180702-112625.csv')#qlMaze.data.20180630-205150.csv') 


##################
mazeToRun <- mazex
upperBound <- 16000

#setwd("/Users/773742/Documents/CEC2018/DST2018/")
raw.data1 <- read.csv(file =   mazeToRun[2] #Train - 201801141417 - Trial 0 - TRIAL_NUM - 6000 - TEST.csv.csv"
                      , header = TRUE, sep = "|"
                      , stringsAsFactors = FALSE
                      , row.names=NULL)


#grepl('chars', 'charsxxx')

raw.data <- raw.data1
 
 
data <- raw.data %>% 
  select(TrailNumber, Timestamp, Matched, MatchedFinal)  %>%
  filter( Timestamp <= upperBound)


################ calculate match rate ###############
result <- data %>%
  group_by(TrailNumber, Timestamp ) %>%
  summarise(groupRow = n()
            , matchCount = sum(Matched)
            , matchRate =matchCount/groupRow 
            , matchLastCount = sum(MatchedFinal)
            , matchLastRate = matchLastCount/groupRow )



################ calculate mean match rate and hyper volume ###############
retdata <- result %>%
  group_by(Timestamp) %>%
  summarise(matchRateAvg = mean(matchRate)  
            , maxmr = max(matchRate)
            , minmr = min(matchRate)
            , matchLastRateAvg = mean(matchLastRate)  
            , maxmlr = max(matchLastRate)
            , minmlr = min(matchLastRate))




################ plot data ###############
plot.data <- retdata 
# %>% filter(TraceWeight  %in% plot.traceWeightFilter
#                                 , Timestamp <= plot.upperBound
#                                 , TargetWeight %in% plot.targetReward
# )


plot.data$matchRateAvg


phv <- ggplot(data = plot.data, aes(
  x = Timestamp,
  y = matchLastRateAvg
)) +
  geom_line() + 
  ylim(0, 1) +
  labs(x = 'Number of Learning Problems\n(a)', y = NULL) +
  ggtitle("% LAST") +
  theme(axis.title.y = element_text(size = rel(1.1), face = "bold"), axis.title.x = element_text(size = rel(1.1), face = "bold"), title = element_text(size = rel(1.1), face = 'bold')) +
  theme(legend.text = element_text(size = rel(1), face = "bold")) +
  theme(legend.title = element_blank()) +
  #theme(legend.position = c(0.63, 0.15))
  theme(legend.position = 'bottom') + theme(panel.grid.major = element_line(size = 0.01, linetype = 'dotted',
                                                                            colour = "black"),
                                            panel.grid.minor = element_line(size = 0.001, linetype = 'dotted',
                                                                            colour = "black")) +
  theme(legend.background = element_rect(fill = alpha('gray', 0.05))) +
  theme(axis.text.x = element_text(size = rel(1.4)),
        axis.text.y = element_text(size = rel(1.4)),
        axis.line.x = element_line(size = rel(0.4),colour = 'black',linetype = 'solid'),
        axis.line.y = element_line(size = rel(0.4),colour = 'black',linetype = 'solid'),
        axis.title = element_text(size = rel(1.2), face = "bold")) 

################ plot match rate ###############
pmr <- ggplot(data = plot.data, aes(
  x = Timestamp,
  y = matchRateAvg )) +
  geom_line() +
  ylim(0, 1) +
  #geom_ribbon(aes(ymin = minmr, ymax = maxmr, fill = TraceWeight), alpha = 0.2) +
  labs(x = 'Number of Learning Problems\n(b)', y = NULL) +
  ggtitle("% OP") +
  theme(axis.title.y = element_text(size = rel(1.1), face = "bold"), axis.title.x = element_text(size = rel(1.1), face = "bold"), title = element_text(size = rel(1.1), face = 'bold')) +
  theme(legend.text = element_text(size = rel(1), face = "bold")) +
  theme(legend.title = element_blank()) +
  #theme(legend.position = c(0.63, 0.15))
  theme(legend.position = 'bottom') + theme(panel.grid.major = element_line(size = 0.01, linetype = 'dotted',
                                                                            colour = "black"),
                                            panel.grid.minor = element_line(size = 0.001, linetype = 'dotted',
                                                                            colour = "black")) +
  theme(legend.background = element_rect(fill = alpha('gray', 0.05))) +
  theme(axis.text.x = element_text(size = rel(1.4)),
        axis.text.y = element_text(size = rel(1.4)),
        axis.line.x = element_line(size = rel(0.4),colour = 'black',linetype = 'solid'),
        axis.line.y = element_line(size = rel(0.4),colour = 'black',linetype = 'solid'),
        axis.title = element_text(size = rel(1.2), face = "bold")) 
  #scale_linetype_manual(values = lty) +
  #scale_colour_manual(values = cbbPalette)



################ plot arrange plots into one ###############
library(gridExtra)


grid.arrange(pmr, phv, nrow=1, ncol=2)