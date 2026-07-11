# 心绪微澜（MoodRipple）

为 AstrBot QQ 机器人添加一个不改变底层人格的情绪层：全局心情、动态情绪标签、每位用户的好感与关系摘要、主动联系和内部情绪日记。

## 安装

将本仓库放入 AstrBot 的 `data/plugins/astrbot_plugin_moodripple`，在插件管理页启用并重载。需要 AstrBot `>=4.5.7,<5`，且已配置可用聊天模型。配置会由 `_conf_schema.json` 自动显示在插件管理页。

运行时状态位于 `data/plugin_data/moodripple/state.json`，不会写入插件源码目录。

## 指令

- `/mood` 或 `/心情`：返回一句诗意化的心情描述，不显示数值。
- `/moodjournal` 或 `/情绪日记`：管理员查看最近一条内部情绪日记。
- `/mooddebug state`：管理员查看内部心情、词条和用户记录数量。
- `/mooddebug labels`：管理员立即让 AI 重生成心情词条。
- `/mooddebug event`：管理员立即生成事件，并在应用事件后重新总结词条。
- `/mooddebug set <值>`：管理员覆盖调试心情值（自动限制在 `-100..100`）。
- `/mooddebug affection <QQ号>`：管理员查询指定用户的好感度。
- `/mooddebug setaffection <QQ号> <值>`：管理员设置指定用户的好感度。
- `/mooddebug relation <QQ号>`：管理员查询指定用户的关系描述。
- `/mooddebug proactive <QQ号>`：管理员立即对已有可用会话的指定用户发起主动消息。
- `/mooddebug journal`：管理员立即重写当天的内部情绪日记。

## 隐私与行为边界

- 对话情绪、每日事件、群聊氛围和主动消息均由配置的 AI 模型评估或生成；插件没有情绪关键词/正则规则。
- 群聊只在机器人近期有回复的群内，按批次向模型发送无昵称的短文本并要求只输出匿名氛围摘要；原始群文本不写入状态文件。
- 主动名单填写 QQ 号。目标用户需要先与机器人交谈一次，插件才有 AstrBot 的 `unified_msg_origin` 可安全发送；近期活跃时只缓存话题，等其下次发言自然引入。
- 主动消息具备概率、冷却和会话窗口限制。请仅将已同意接收机器人消息的用户加入名单。

## 设计说明

每日随机事件在已配置时间窗口内随机排程；每日衰减是唯一非 AI 的心情变化，它按配置把数值向中性回归。好感度的原始变化完全由 AI 给出，再以边界阻尼曲线应用，使其在中段更敏感、临近 `-100/100` 时变缓。

事件生成会读取 AstrBot 当前默认人格的名称和系统提示词，只在本次 AI 请求中使用、不写入插件状态；AI 必须产出带场景、动作与感官细节的具体虚构小事。

启用 `enable_qq_status_sync` 后，插件会在随机事件后生成诗意描述，并最佳努力调用已加载 QQ 适配器显式提供的 `set_bot_status` 接口写入签名/状态；标准 AstrBot 接口并未统一这一能力，因此未提供该接口的适配器会被安全跳过并记录一次日志。

## 验证

```powershell
python -m unittest discover -s tests -v
python -m compileall main.py moodripple
```
