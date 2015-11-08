//
//  Tuser.hpp
//  nann
//
//  Created by logic.yan on 15/11/6.
//  Copyright © 2015年 nagain. All rights reserved.
//

#ifndef Tuser_hpp
#define Tuser_hpp

#include <stdio.h>

class Tuser {
public:
  Tuser();
  virtual ~Tuser();
  
public:
  int _login_time;  /* [0, 86400] */
  int _action_time; /* [0, 3600] */
  int _touch_num;   /* [4, 20] */
  int _os_version;  /* [30,40,41,42,43,44,50,55,60] */
  int _gps;         /* 10, 20, 30, 40, 50+ */
};



#endif /* Tuser_hpp */
