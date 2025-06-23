import logging

# 配置日志记录
def setup_multi_view_logger(log_file="./output/multi_view.log"):
    """
    设置日志记录器，默认写入指定的日志文件。
    默认level为DEBUG，输出格式为时间戳、日志名称、日志级别和消息内容。
    """
    logger = logging.getLogger("multi_view_logger")
    logger.setLevel(logging.DEBUG)  # 设置日志级别为INFO及以上
    
    # 确保不会有重复的处理器
    # 默认是没有处理器的, 不添加则不会输出到任何地方
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
    
    return logger

# 使用示例
if __name__ == "__main__":
    # 设置日志记录器，默认写入process.log文件
    logger = setup_multi_view_logger()
    
    # 记录不同级别的日志
    logger.debug("这是调试信息，不会被记录")
    logger.info("这是常规信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
    
    # 记录变量或计算结果
    result = 42 * 100
    logger.info(f"计算结果: {result}")
    
    # 记录异常信息
    try:
        1 / 0
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
