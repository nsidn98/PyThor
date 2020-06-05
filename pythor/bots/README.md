# Telegrad Robots

This folder contains the robots used to send updates on Telegram.

The original repository can be found [here](https://github.com/eyalzk/telegrad).

## Set-up
* First install the Telegram application on the mobile.
* Follow this [blog](https://towardsdatascience.com/how-to-monitor-and-control-deep-learning-experiments-through-your-phone-35ef1704928d) by the original author of Telegrad to make a bot on Telegram.
* Get the bot token.
* Get your telegram user ID:
    * Search @userinfobot in the search bar in Telegram.
    * Type `/start` to get your user ID.

Fill in your user ID and bot token in [`config.py`](https://github.com/nsidn98/PyThor/blob/master/pythor/bots/config.py)

```

telegram_token = "TOKEN"  # replace TOKEN with your bot's token

#  user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = None  # replace None with your telegram user id (integer)

telegram_config = {'token':telegram_token, 'user_id':telegram_user_id}
```

Now you are ready to run your code from the [PyThor](https://github.com/nsidn98/PyThor) folder.

## Contents:
* `botCallback.py` is the callback function used with pytorch lightning for deep-learning experiments.
* `dl_bot.py` is the bot definition for any deep-learning experiments.
* `rlCallback.py` is the callback function used with pytorch lightning for reinforcement learning experiments.
* `rl_bot.py` is the bot definition for any reinforcement-learning experiments.
