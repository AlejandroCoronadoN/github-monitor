import {useState} from "react"
import './ItemIterator.css';

const ItemDetails = ({
    id,
    key,
    item,
    handleSuggestionClick
}) => {

    const [isSelected, setIsSelected] = useState(false);


  return (
    <div className="item-suggestion" onClick={() => handleSuggestionClick(item)}>
        <span className="author">{item.author + "/"}</span>
        <span className="repo">{item.repository}</span>

    </div>
  )
}



export default ItemDetails
