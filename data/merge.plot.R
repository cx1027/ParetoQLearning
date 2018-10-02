library(dplyr)
library(stringr)
library(ggplot2)


############ settings ##############
mmaze.xcs <- c('xcs','xcs.mmaze_path.csv','mmaze.XCS_FINAL_NMAZE.csv')
mmaze.zcs <- c('zcs', 'xcs.mmaze_path.csv','mmaze.zcs.nxcs.testbed.dst_weighted_sum - 201809252124 - Trial 0 - TRIAL_NUM - 150000 - TEST.csv')
mmaze.pql <-  c('pql', 'xcs.mmaze_path.csv','nmaze.pyqlMaze.data.20180919-222357x.csv')

mmaze.xcs <- c('xcs','xcs.mmaze_path.csv','mmaze - nxcs.testbed.mmaze_weighted_sum - 20181002144152 - Trial 0 - TRIAL_NUM - 150000 - TEST.csv')
mmaze.pql <-  c('pql', 'xcs.mmaze_path.csv','nmaze.pyqlMaze.data.20181002-133445.csv')
mmaze.xcs <- c('xcs','xcs.mmaze_path.csv','mmaze - nxcs.testbed.mmaze_weighted_sum - 20181002230525 - Trial 0 - TRIAL_NUM - 150000 - TEST.csv')

mazes <- list(mmaze.xcs, mmaze.zcs, mmaze.pql)

mazeToRun <- mmaze.xcs #mazes[2]




ref.point <- c(-30, -30)
plot.upperbond <- 500
TraceWeight.filter <- c('0.000000|1.000000', 
                        '0.100000|0.900000', 
                        #'0.200000|0.800000', 
                        #'0.300000|0.700000', 
                        '0.400000|0.600000',
                        #'0.500000|0.500000', 
                        '0.600000|0.400000', 
                        #'0.700000|0.300000', 
                        #'0.800000|0.200000', 
                        '0.900000|0.100000', 
                        '1.000000|0.000000'
)


getprepdata <-    function(mazeToRun){
  
  upperBound <- 6000
  
  ############# begin to read result #############
  setwd("C:/Users/martin.xie/IdeaProjects/XCS_MOEAD/dataAnalysis/Maze")
  
  targetSteps <- read.csv(file = mazeToRun[2], header = TRUE, sep = ",", stringsAsFactors = FALSE)
  targetId <- paste(targetSteps$open, targetSteps$final, paste(as.character(targetSteps$step), '', sep = ''), sep = '*')
  targetSteps <- cbind(targetSteps, targetId)
  
  reward.file <- 'mmaze.rewards.csv'
  maze.reward <- read.csv(file = reward.file, header = TRUE, sep = ",", stringsAsFactors = FALSE)
  
  
  setwd("C:/Temp/Results")
  sepx <- ','
  if(mazeToRun[1] =='pql'){
    sepx <- '|'
  }
  raw.data <- read.csv(file =   mazeToRun[3] #Train - 201801141417 - Trial 0 - TRIAL_NUM - 6000 - TEST.csv.csv"
                       , header = TRUE, sep =  sepx
                       , stringsAsFactors = FALSE
                       , row.names=NULL)
  
  
  #h5data <- head(raw.data,10000)
  if(mazeToRun[1] =='pql'){
    
    raw.data$TargetWeight <- '30.000000|20.000000'
    raw.data$TraceWeight <- TraceWeight.filter[1]
    data <- raw.data  %>% #filter (TargetWeight=='100.000000|30.000000') %>% 
      select(TrailNumber, Timestamp, TargetWeight, TraceWeight, OpenState, FinalState, steps, hyperVolumn, match) %>%
      filter(#TraceWeight %in% traceWeightFilter
        Timestamp <= upperBound)
    data$FinalState <- gsub(', ', '-', data$FinalState)
  }
  
  
  
  
  
  if(mazeToRun[1] == 'zcs'){
    data <- raw.data  %>% #filter (TargetWeight=='100.000000|30.000000') %>% 
      select(TrailNumber, Timestamp, TargetWeight, TraceWeight, OpenState, FinalState, steps, hyperVolumn) %>%
      filter(#TraceWeight %in% traceWeightFilter
        Timestamp <= upperBound)
    ################ check if uid in final state pair ###############
    uid <- paste(data$OpenState, data$FinalState, data$steps, sep = "*")
    data$uid <- uid
    data$match <- ifelse(data$uid %in% targetSteps$targetId, 1, 0)
  }
  if(mazeToRun[1] == 'xcs'){
    data <- raw.data  %>% #filter (TargetWeight=='100.000000|30.000000') %>% 
      select(TrailNumber, Timestamp, TargetWeight, TraceWeight, OpenState, FinalState, steps, hyperVolumn, matched) %>%
      filter(#TraceWeight %in% traceWeightFilter
        Timestamp <= upperBound)
    ################ check if uid in final state pair ###############
    
    data$match <- data$matched
  }
  
  
  
  data$resteps <- ifelse(data$match == 1, -data$steps, -30)
  
  
  data.x <- merge(data, maze.reward, by = c("FinalState"), all.x = TRUE)
  data.filtered <- data.x #%>% filter(TraceWeight %in%  c('0.000000|1.000000', '1.000000|0.000000' ))
  
  #TrailNumber, Timestamp, TargetWeight, TraceWeight, OpenState, FinalState, steps, 
  data.ordered <- data.filtered[
    with(data.filtered, order(TrailNumber, Timestamp, TargetWeight, OpenState, resteps)),
    ]
  
  
  
  if(mazeToRun[1] != 'pql'){
  
    nweight <- length(unique(data$TraceWeight))
    
    
    sel <- rep(FALSE, nweight)
    i <- 1
    subsel.list <- list()
    for (i in 1:nweight) {
      
      subsel <- sel
      subsel[i] = T
      subsel.list[[i]] <- subsel
      
    }
    i <- 0
    ret.data <- NULL
    for (li in subsel.list){
        tmp <- data.ordered[li,]
        #tmp$hv <- 0
        if(i == 0){
          tmp$hv <- abs(tmp$resteps - ref.point[1] ) * abs(tmp$reward - ref.point[1])
          ret.data <- tmp
          prev.data <- tmp
        }else
        {
          tmp$hv <- abs(tmp$resteps -  prev.data$resteps ) * abs(tmp$reward - prev.data$reward)
          ret.data <- rbind(ret.data, tmp)
        }
        print(li)
        i = i+1
    }
    # li2 <- subsel.list[[2]]
    # li3 <- subsel.list[[3]]
    # data.li2 <- data.ordered[li2,]
    # data.li3 <- data.ordered[li3,]
    
    # tmp$hv <- abs(tmp$resteps -  prev.data$resteps ) * abs(tmp$reward - prev.data$reward)
    # ret.data <- rbind(ret.data, tmp)
  
  }else{
    data$hv <- data$hyperVolumn
    ret.data <- data
  }
  
  #TrailNumber, Timestamp, TargetWeight, TraceWeight, OpenState, FinalState, steps, 
  data.hv <- ret.data[
    with(ret.data, order(TrailNumber, Timestamp, TargetWeight, OpenState, resteps)),
    ]
  
  return (data.hv)  
}

data.xcs <- getprepdata(mmaze.xcs)
data.zcs <- getprepdata(mmaze.zcs)
data.pql <- getprepdata(mmaze.pql)
data.xcs$src <- 'xcs'
data.zcs$src <- 'zcs'
data.pql$src <- 'pql'


predata<- data.pql

getresult <- function(predata) {
################ calculate match rate ###############
  result.hv <- predata %>%
    group_by(TrailNumber, Timestamp,TargetWeight ) %>%
    summarise(groupRow = n()
              , matchCount = sum(match)
              , matchRate = matchCount/groupRow 
              , hv = sum(hv))
  if(predata$src[1] == 'pql')
    result.hv <- predata %>%
      group_by(TrailNumber, Timestamp,TargetWeight ) %>%
      summarise(groupRow = n()
                , matchCount = sum(match)
                , matchRate = matchCount/groupRow 
                , hv = mean(hv))
  
  ################ calculate mean match rate and hyper volume ###############
  retdata.hv <- result.hv %>%
    group_by(Timestamp, TargetWeight) %>%
    summarise(#matchRateAvg = mean(matchRate) 
               hyperVolumnAvg = mean(hv)
              , maxhv = max(hv)
              , minhv = min(hv))
  
  
  
  result.mr <- predata %>%
    group_by(TrailNumber, Timestamp,TargetWeight,TraceWeight ) %>%
    summarise(groupRow = n()
              , matchCount = sum(match)
              , matchRate = matchCount/groupRow 
    )
  
  
  
  ################ calculate mean match rate and hyper volume ###############
  retdata.mr <- result.mr %>%
    group_by(Timestamp, TargetWeight,TraceWeight) %>%
    summarise(matchRateAvg = mean(matchRate) 
              , maxmr = max(matchRate)
              , minmr = min(matchRate))
  

  return (list( mr = retdata.mr, hv = retdata.hv))
}

plot.data.xcs <- getresult(data.xcs)
plot.data.zcs <- getresult(data.zcs)
plot.data.pql <- getresult(data.pql)

plot.data.xcs[[1]]$src <- 'xcs'
plot.data.zcs[[1]]$src <- 'zcs'
plot.data.pql[[1]]$src <- 'pql'

#padding trace weight for data.pql
plot.data.pql.org <- plot.data.pql[[1]]
plot.data.pql[[1]] <- NULL
for (x in unique(plot.data.xcs[[1]]$TraceWeight)) {
  plot.data.pql.org$TraceWeight <- x
  plot.data.pql[[1]] <- rbind(plot.data.pql[[1]], plot.data.pql.org)
}

plot.data.xcs[[2]]$src <- 'xcs'
plot.data.zcs[[2]]$src <- 'zcs'
plot.data.pql[[2]]$src <- 'pql'


plot.data.merge.mr <- rbind(plot.data.xcs[[1]], plot.data.zcs[[1]], plot.data.pql[[1]])
plot.data.merge.hv <- rbind(plot.data.xcs[[2]], plot.data.zcs[[2]], plot.data.pql[[2]])

#plot.data.merge.mr <- plot.data.merge.mr %>% filter(Timestamp <= plot.upperbond, TraceWeight %in% TraceWeight.filter)
plot.data.merge.mr <- plot.data.merge.mr %>% filter(Timestamp <= plot.upperbond, TraceWeight %in% TraceWeight.filter)
plot.data.merge.hv <- plot.data.merge.hv %>% filter(Timestamp <= plot.upperbond)


plot.merge.xcs <- ggplot(data = plot.data.merge.mr %>% filter(src == 'xcs'), aes(
  x = Timestamp,
  y = matchRateAvg,
  colour = src,
  group = src,
  linetype = src)) +
  geom_line() +
  #geom_ribbon(aes(ymin = minmr, ymax = maxmr, fill = TraceWeight), alpha = 0.2) +
  labs(x = 'Number of Learning Problems\n(a)', y = NULL) +
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


plot.merge.mr.m <- plot.merge.xcs + 
  geom_line(data = plot.data.merge.mr %>% filter(src == 'zcs'), aes(
    x = Timestamp,
    y = matchRateAvg,
    colour = src,
    group = src,
    linetype = src))  


plot.merge.mr.m2 <- plot.merge.mr.m + 
  geom_line(data = plot.data.merge.mr %>% filter(src == 'pql'), aes(
    x = Timestamp,
    y = matchRateAvg,
    colour = src,
    group = src,
    linetype = src))  

plot.merge.mr.m2

plot.merge.mr.m2 + facet_grid(TraceWeight ~ .)
plot.merge.mr.m2 + facet_grid(.~ TraceWeight )

#------------------------------------------------------

plot.merge.xcs.hv <- ggplot(data = plot.data.merge.hv %>% filter(src == 'xcs'), aes(
  x = Timestamp,
  y = hyperVolumnAvg,
  colour = src,
  group = src,
  linetype = src)) +
  geom_line() +
  #geom_ribbon(aes(ymin = minmr, ymax = maxmr, fill = TraceWeight), alpha = 0.2) +
  labs(x = 'Number of Learning Problems\n(b)', y = NULL) +
  ggtitle("THV") +
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


plot.merge.hv.m <- plot.merge.xcs.hv + 
  geom_line(data = plot.data.merge.hv %>% filter(src == 'zcs'), aes(
    x = Timestamp,
    y = hyperVolumnAvg,
    colour = src,
    group = src,
    linetype = src))  


plot.merge.hv.m2 <- plot.merge.hv.m + 
  geom_line(data = plot.data.merge.hv %>% filter(src == 'pql'), aes(
    x = Timestamp,
    y = hyperVolumnAvg,
    colour = src,
    group = src,
    linetype = src))  

plot.merge.hv.m2






################ plot arrange plots into one ###############
library(gridExtra)

g_legend <- function(a.gplot) {
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

mylegend <- g_legend(plot.merge.hv.m2)

p5.a <- grid.arrange(arrangeGrob(
  plot.merge.mr.m2 + theme(legend.position = "none"),
  plot.merge.hv.m2 + theme(legend.position = "none"),
  nrow = 1),
  mylegend, nrow = 2, heights = c(5, 1) )

p5.a


